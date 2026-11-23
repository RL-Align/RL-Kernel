# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Packed ``(t, h, w)`` rotary coordinates of MiniMax-H3-Image (RFC #738 ``image_position_grid``).

The released pipeline lays one request out as ``[text | reference 1 | ... | reference N |
generated image]`` and gives every row a float64 ``(t, h, w)`` coordinate:

* text row ``i`` sits at ``(i, 0, 0)``;
* every image is one frame on the rotary clock. The clock starts where the text rows end,
  each reference takes one integer slot in packed order, and the generated image takes the
  slot after the last reference;
* inside an image, ``(h, w)`` is the row-major mesh of two aspect-normalised axes. For a
  latent side ``d``, patch ``p`` and ``sqrt_area = sqrt(latent_h * latent_w)`` of *that*
  image::

      ratio = d / sqrt_area
      left  = (1 - ratio) / 2
      axis  = numpy.linspace(left, left + ratio, d // p, endpoint=False) * 32

  ``numpy.linspace(endpoint=False)`` computes ``start + arange(n) * step``, which is not
  what ``torch.linspace`` computes, so the grid is built with NumPy in float64. Each
  reference keeps its own geometry; nothing is resized to the target grid.

Upstream: ``build_image_packed_sequence`` / ``_frame_position_grid`` /
``_spatial_position_grid`` in Diffusers ``modular_pipelines/minimax_h3_image/before_denoise.py``
at commit 875373ac7f5856071e345a3e394c74efccc6b9e6 (huggingface/diffusers#15230). The spatial
axis is the same construction the MiniMax-H3 video path uses (RFC #420), with the image
clock replacing the video frame clock.

The transformer's rotary module casts the coordinates to float32 before multiplying by
``inv_freq``. A sealed artifact therefore records the float64 grid and its float32 cast,
each with its own digest (the RFC #420 trajectory ``tensor_digest``: dtype, shape, raw
bytes), so a drift is attributed to grid construction or to the cast.

``torch_linspace_axis`` is the I8 probe arm: the mathematically similar Torch
construction. It is a different comparator profile and never used by the strict path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from rl_engine.models.minimax_h3.trajectory import tensor_digest

SPATIAL_SCALE = 32
DEFAULT_PATCH = (1, 2, 2)
MAX_REFERENCES = 16
CONSTRUCTION = "numpy.linspace(left, left + d / sqrt(h * w), d // p, endpoint=False) * 32, float64"
SCHEMA_VERSION = "rlkernel.h3_image.position_grid.v1"


def _check_patch(patch_size: Sequence[int]) -> tuple[int, int]:
    if len(patch_size) != 3 or any(int(p) <= 0 for p in patch_size):
        raise ValueError(f"patch_size must be three positive ints, got {tuple(patch_size)}")
    if int(patch_size[0]) != 1:
        raise ValueError("an image is a single frame: the temporal patch must be 1")
    return int(patch_size[1]), int(patch_size[2])


def _check_grid(name: str, latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError(
            f"{name}: latent size must be positive, got {latent_height}x{latent_width}"
        )
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError(
            f"{name}: latent {latent_height}x{latent_width} is not divisible by the "
            f"{patch_h}x{patch_w} patch"
        )


def spatial_axis(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    """One aspect-normalised spatial axis, float64, exactly as the released pipeline builds it."""

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * SPATIAL_SCALE


def torch_linspace_axis(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """I8 probe arm: the same axis through ``torch.linspace`` (float64). Not the strict path."""

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    count = dim // patch
    stop = left + ratio
    # endpoint=False: torch.linspace includes its end, so stop one step early.
    grid = torch.linspace(left, stop - (stop - left) / count, count, dtype=torch.float64)
    return grid * SPATIAL_SCALE


def frame_grid(
    latent_height: int,
    latent_width: int,
    patch_size: Sequence[int] = DEFAULT_PATCH,
    *,
    axis: str = "numpy",
) -> torch.Tensor:
    """Row-major ``(h, w)`` coordinates of one image, float64 ``[rows, 2]``."""

    patch_h, patch_w = _check_patch(patch_size)
    _check_grid("image", latent_height, latent_width, patch_h, patch_w)
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    if axis == "numpy":
        height = torch.from_numpy(spatial_axis(latent_height, patch_h, sqrt_area))
        width = torch.from_numpy(spatial_axis(latent_width, patch_w, sqrt_area))
    elif axis == "torch":
        height = torch_linspace_axis(latent_height, patch_h, sqrt_area)
        width = torch_linspace_axis(latent_width, patch_w, sqrt_area)
    else:
        raise ValueError(f"axis must be 'numpy' or 'torch', got {axis!r}")
    grids = torch.meshgrid(height.to(torch.float64), width.to(torch.float64), indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1)


@dataclass(frozen=True)
class ImageSource:
    """Where one image's rows sit in the packed sequence."""

    kind: str  # "reference" or "target"
    index: int  # reference index, or -1 for the target
    latent_height: int
    latent_width: int
    rotary_time: float
    start: int
    stop: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "index": self.index,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "rotary_time": self.rotary_time,
            "start": self.start,
            "stop": self.stop,
        }


@dataclass(frozen=True)
class PackedPositions:
    """``position_ids`` of one packed layout plus the row map it was built from."""

    position_ids: torch.Tensor  # float64 [S, 3]
    num_text_tokens: int
    sources: tuple[ImageSource, ...]
    axis: str

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def num_reference_rows(self) -> int:
        return sum(s.stop - s.start for s in self.sources if s.kind == "reference")

    def artifact(self) -> dict[str, Any]:
        """Sealed, machine-readable description of this grid (RFC #738 I-C4)."""

        return {
            "schema_version": SCHEMA_VERSION,
            "construction": CONSTRUCTION if self.axis == "numpy" else f"probe:{self.axis}",
            "dtype": "float64",
            "shape": list(self.position_ids.shape),
            "num_text_tokens": self.num_text_tokens,
            "num_reference_rows": self.num_reference_rows,
            "sources": [s.to_dict() for s in self.sources],
            "sha256_fp64": tensor_digest(self.position_ids),
            "sha256_fp32": tensor_digest(self.position_ids.to(torch.float32)),
        }


def packed_positions(
    num_text_tokens: int,
    reference_grids: Sequence[tuple[int, int]],
    target_grid: tuple[int, int],
    patch_size: Sequence[int] = DEFAULT_PATCH,
    *,
    max_references: int = MAX_REFERENCES,
    axis: str = "numpy",
) -> PackedPositions:
    """Build float64 ``position_ids`` for ``[text | references | target]``.

    ``reference_grids`` and ``target_grid`` are latent ``(height, width)`` sizes, in packed
    order. Every geometry is checked before any coordinate is built; an unsupported layout
    raises instead of producing a partial grid.
    """

    patch_h, patch_w = _check_patch(patch_size)
    if isinstance(num_text_tokens, bool) or int(num_text_tokens) != num_text_tokens:
        raise ValueError("num_text_tokens must be an int")
    if num_text_tokens < 0:
        raise ValueError(f"num_text_tokens must be >= 0, got {num_text_tokens}")
    if len(reference_grids) > max_references:
        raise ValueError(f"{len(reference_grids)} references exceed the limit of {max_references}")
    grids = [(int(h), int(w)) for h, w in reference_grids]
    for index, (height, width) in enumerate(grids):
        _check_grid(f"reference {index}", height, width, patch_h, patch_w)
    target = (int(target_grid[0]), int(target_grid[1]))
    _check_grid("target", target[0], target[1], patch_h, patch_w)

    rows = [(h // patch_h) * (w // patch_w) for h, w in grids + [target]]
    sequence_length = num_text_tokens + sum(rows)
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    sources: list[ImageSource] = []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for index, ((height, width), count) in enumerate(zip(grids + [target], rows)):
        is_target = index == len(grids)
        position_ids[cursor : cursor + count, 0] = rotary_time
        position_ids[cursor : cursor + count, 1:] = frame_grid(height, width, patch_size, axis=axis)
        sources.append(
            ImageSource(
                "target" if is_target else "reference",
                -1 if is_target else index,
                height,
                width,
                rotary_time,
                cursor,
                cursor + count,
            )
        )
        cursor += count
        rotary_time += 1.0
    return PackedPositions(position_ids, int(num_text_tokens), tuple(sources), axis)


def first_drift(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    """Locate the first differing ``(row, axis)`` of two position tensors, or None if equal.

    Accepts packed ``[S, 3]`` (``t, h, w``) or frame ``[rows, 2]`` (``h, w``) coordinates.
    """

    if a.shape != b.shape:
        return {"reason": "shape", "a": list(a.shape), "b": list(b.shape)}
    diff = (a != b).nonzero()
    if diff.numel() == 0:
        return None
    names = {3: "thw", 2: "hw"}.get(int(a.shape[-1]))
    if a.ndim != 2 or names is None:
        raise ValueError(f"expected [rows, 3] or [rows, 2] coordinates, got {list(a.shape)}")
    row, column = (int(v) for v in diff[0])
    return {
        "reason": "value",
        "row": row,
        "axis": names[column],
        "a": float(a[row, column]),
        "b": float(b[row, column]),
        "differing_values": int(diff.shape[0]),
        "max_abs": float((a - b).abs().max()),
    }
