# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Image latent patch pack / unpack of MiniMax-H3-Image (RFC #738 ``image_patch_pack_unpack``).

An image latent is one VAE frame, ``(B, 24, H, W)``. The transformer reads it as rows of
width ``24 * 1 * 2 * 2 = 96``, frame-major then row-major over the ``(H/2, W/2)`` patch grid,
each row channel-major then ``(ph, pw)``. That is the MiniMax-H3 video permutation (RFC #420
``video_patch_pack_unpack``) with ``T = 1``, so this module does not reimplement it: it
calls the registered H3 backend (``pytorch-video-patch-pack-unpack-v1`` by default, or any
op with the same ``pack`` / ``unpack`` interface) and adds what the image path needs:

* ``pack_image``: ``(B, 24, H, W)`` or ``(B, 24, 1, H, W)`` -> ``(B, H/2 * W/2, 96)``;
* ``pack_references``: each edit reference keeps its own geometry and the rows are
  concatenated in packed order, as the released prepare-references step does;
* ``unpack_target``: only the generated suffix of the image selection (after
  ``num_reference_rows``) goes back to ``(B, 24, H, W)``; the reference rows are context
  and are never unpacked;
* ``patch_of_row``: the latent block a row came from, matching ``packed_layout``'s
  ``patch_coord``.

The permutation is pure data movement: no arithmetic, no dtype conversion, so pack and
unpack are exact inverses bit for bit. Geometry the image path does not support raises
before any data is moved: wrong rank, channels, ``T != 1``, odd latent sides, a row count
that does not match the latent grid, or a reference count that does not match the layout.

Upstream: ``patchify_video_latents`` (``before_denoise.py``) and the unpatchify in
``MiniMaxH3ImageAfterDenoiseStep`` (``decoders.py``) of Diffusers
``modular_pipelines/minimax_h3_image`` at commit 875373ac7f5856071e345a3e394c74efccc6b9e6.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from rl_engine.reference.packing.video_patch_pack_unpack import NativeVideoPatchPackUnpackOp

CHANNELS = 24
PATCH = (1, 2, 2)
WIDTH = CHANNELS * PATCH[0] * PATCH[1] * PATCH[2]


class ImagePatchPacker:
    """Image-shaped adapter over the H3 ``video_patch_pack_unpack`` backend."""

    def __init__(self, op: Any | None = None) -> None:
        self.op = op if op is not None else NativeVideoPatchPackUnpackOp()

    @property
    def backend_id(self) -> str:
        return str(getattr(self.op, "backend_id", type(self.op).__name__))

    # --- pack -----------------------------------------------------------------------------

    def pack_image(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, 24, H, W)`` or ``(B, 24, 1, H, W)`` -> rows ``(B, H/2 * W/2, 96)``."""

        frame = _as_frame(latents, "image latents")
        return self.op.pack(frame)

    def pack_references(self, references: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concatenate the rows of every reference, in packed order: ``(B, R, 96)``.

        Each reference is ``(B, 24, h_i, w_i)`` with its own geometry; all share ``B`` and
        dtype. An empty list (text-to-image) is an error: there are no reference rows.
        """

        if not references:
            raise ValueError("pack_references needs at least one reference")
        rows = [self.pack_image(reference) for reference in references]
        batch = {int(r.shape[0]) for r in rows}
        dtypes = {r.dtype for r in rows}
        if len(batch) != 1 or len(dtypes) != 1:
            raise ValueError("references must share batch size and dtype")
        return torch.cat(rows, dim=1)

    # --- unpack ---------------------------------------------------------------------------

    def unpack_image(self, rows: torch.Tensor, latent_hw: tuple[int, int]) -> torch.Tensor:
        """Rows ``(B, H/2 * W/2, 96)`` -> ``(B, 24, H, W)``."""

        height, width = _check_hw(latent_hw)
        if rows.ndim != 3:
            raise ValueError(f"rows must be (B, S, {WIDTH}), got {list(rows.shape)}")
        shape = (int(rows.shape[0]), CHANNELS, 1, height, width)
        return self.op.unpack(rows, shape)[:, :, 0]

    def unpack_target(
        self, image_rows: torch.Tensor, num_reference_rows: int, latent_hw: tuple[int, int]
    ) -> torch.Tensor:
        """Unpack the generated suffix of the image selection: ``(B, 24, H, W)``."""

        if isinstance(num_reference_rows, bool) or num_reference_rows < 0:
            raise ValueError("num_reference_rows must be a non-negative int")
        height, width = _check_hw(latent_hw)
        if image_rows.ndim != 3:
            raise ValueError(f"image rows must be (B, N, {WIDTH}), got {list(image_rows.shape)}")
        target_rows = (height // PATCH[1]) * (width // PATCH[2])
        if int(image_rows.shape[1]) != num_reference_rows + target_rows:
            raise ValueError(
                f"{int(image_rows.shape[1])} image rows != {num_reference_rows} reference + "
                f"{target_rows} target rows for a {height}x{width} latent"
            )
        return self.unpack_image(image_rows[:, num_reference_rows:].contiguous(), (height, width))

    # --- metadata -------------------------------------------------------------------------

    def provenance(self) -> dict[str, Any]:
        return {
            "semantic_op": "video_patch_pack_unpack",
            "backend_id": self.backend_id,
            "image_adapter": "rlkernel.h3_image.patch_pack.v1",
            "patch_size": list(PATCH),
            "width": WIDTH,
        }


def patch_of_row(row: int, latent_hw: tuple[int, int]) -> tuple[int, int]:
    """``(patch_row, patch_col)`` of row ``row`` of one image; same as ``patch_coord``."""

    height, width = _check_hw(latent_hw)
    cols = width // PATCH[2]
    rows = height // PATCH[1]
    if not 0 <= row < rows * cols:
        raise ValueError(f"row {row} is outside the {rows}x{cols} patch grid")
    return row // cols, row % cols


def _check_hw(latent_hw: tuple[int, int]) -> tuple[int, int]:
    if len(latent_hw) != 2:
        raise ValueError(f"latent_hw must be (height, width), got {latent_hw!r}")
    height, width = (int(v) for v in latent_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"latent size must be positive, got {height}x{width}")
    if height % PATCH[1] or width % PATCH[2]:
        raise ValueError(f"latent {height}x{width} is not divisible by the 2x2 spatial patch")
    return height, width


def _as_frame(latents: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if latents.ndim == 4:
        latents = latents.unsqueeze(2)
    elif latents.ndim != 5:
        raise ValueError(
            f"{name} must be (B, 24, H, W) or (B, 24, 1, H, W), got {list(latents.shape)}"
        )
    if int(latents.shape[2]) != 1:
        raise ValueError(f"{name} must be a single frame (T = 1), got T = {int(latents.shape[2])}")
    if int(latents.shape[1]) != CHANNELS:
        raise ValueError(f"{name} need {CHANNELS} channels, got {int(latents.shape[1])}")
    _check_hw((int(latents.shape[3]), int(latents.shape[4])))
    return latents
