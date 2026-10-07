# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 AdaLN row gather (RFC #420 row ``adaln_row_gather``).

H3 computes one row index per packed-sequence position and uses it to pick
every position's six modulation vectors::

    adaln_indices = timestep_indices * 3 + token_tags       (S,)
    shift_msa[adaln_indices], scale_msa[adaln_indices], ...  six (S, H)

``rows`` is the ``(3T, 6H)`` view of one block's AdaLN projection
(``table.view(-1, 6H)``); its six column blocks are the six modulation
tensors. ``timestep_indices`` and ``token_tags`` are semantic inputs
(RFC #420 §4): out-of-range values are an error, never clamped.
"""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.pytorch.h3 import H3_ADALN_CHUNKS, H3_MODALITY_NUM

_INDEX_DTYPES = (torch.int64, torch.int32)


def validate_h3_adaln_row_gather(
    rows: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    *,
    check_range: bool = True,
) -> int:
    """Return the hidden size; raise on anything H3 does not define."""

    if not isinstance(rows, torch.Tensor) or rows.dim() != 2:
        raise ValueError("rows must be a 2-D (3T, 6H) tensor")
    if not rows.is_floating_point():
        raise TypeError(f"rows must be floating point, got {rows.dtype}")
    num_rows, width = rows.shape
    if num_rows == 0 or num_rows % H3_MODALITY_NUM != 0:
        raise ValueError(f"rows must hold 3 rows per timestep, got {num_rows}")
    if width == 0 or width % H3_ADALN_CHUNKS != 0:
        raise ValueError(f"rows width must be 6 * H, got {width}")
    for name, index in (("timestep_indices", timestep_indices), ("token_tags", token_tags)):
        if not isinstance(index, torch.Tensor) or index.dim() != 1:
            raise ValueError(f"{name} must be a 1-D (S,) tensor")
        if index.dtype not in _INDEX_DTYPES:
            raise TypeError(f"{name} must be int64 or int32, got {index.dtype}")
        if index.device != rows.device:
            raise ValueError(f"{name} is on {index.device}, rows are on {rows.device}")
    if timestep_indices.shape != token_tags.shape:
        raise ValueError(
            f"timestep_indices {tuple(timestep_indices.shape)} and token_tags "
            f"{tuple(token_tags.shape)} must have the same length"
        )
    if timestep_indices.numel() == 0:
        raise ValueError("the packed sequence must not be empty")
    if timestep_indices.dtype != token_tags.dtype:
        raise TypeError("timestep_indices and token_tags must share an integer dtype")
    if check_range:
        num_timesteps = num_rows // H3_MODALITY_NUM
        bad = (
            (timestep_indices < 0)
            | (timestep_indices >= num_timesteps)
            | (token_tags < 0)
            | (token_tags >= H3_MODALITY_NUM)
        )
        if bool(bad.any()):  # one host sync
            raise IndexError(
                f"timestep_indices must lie in [0, {num_timesteps}) and token_tags in "
                f"[0, {H3_MODALITY_NUM}) (0 video, 1 text, 2 audio)"
            )
    return width // H3_ADALN_CHUNKS


def h3_adaln_indices(timestep_indices: torch.Tensor, token_tags: torch.Tensor) -> torch.Tensor:
    return timestep_indices.long() * H3_MODALITY_NUM + token_tags.long()


class _DeterministicRowGather(torch.autograd.Function):
    """``index_select`` forward; FP32 per-row segment sums for the backward.

    ``index_select``'s own backward is an atomic scatter-add in the grad dtype
    (BF16 for H3), which is neither deterministic nor accurate. There are
    only 3T table rows, so the reference sums each row's positions with one
    FP32 ``torch.sum`` per row (a fixed reduction for a fixed packing) and
    rounds once.
    """

    @staticmethod
    def forward(ctx, rows, index):
        ctx.save_for_backward(index)
        ctx.rows_meta = (rows.shape[0], rows.dtype)
        return torch.stack(
            [chunk.index_select(0, index) for chunk in rows.chunk(H3_ADALN_CHUNKS, dim=-1)]
        )

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        num_rows, dtype = ctx.rows_meta
        grad32 = grad.float()  # (6, S, H)
        out = grad32.new_zeros((num_rows, grad32.shape[0], grad32.shape[2]))
        for row in range(num_rows):
            positions = torch.nonzero(index == row).flatten()
            if positions.numel():
                out[row] = grad32.index_select(1, positions).sum(dim=1)
        return out.reshape(num_rows, -1).to(dtype), None


class NativeH3AdaLNRowGatherOp:
    """PyTorch reference: ``index_select`` on each of the six column blocks.

    The forward is byte-identical to diffusers; the backward is deterministic
    (see ``_DeterministicRowGather``). The raw diffusers path, including its
    atomic backward, is ``rl_engine.testing.h3_provider.provider_adaln_row_gather``.
    ``forward_fp32`` returns the same rows in FP32 with an FP64 gradient.

    The forward is a copy, but the VJP is a segmented reduction over the
    packed sequence, so the op is judged with the ``reduction`` tolerance
    class; forward bitwise equality is asserted separately.
    """

    op_class = "reduction"

    def __call__(self, rows, timestep_indices, token_tags):
        return self.forward(rows, timestep_indices, token_tags)

    def forward(self, rows, timestep_indices, token_tags, *, check_range: bool = True):
        validate_h3_adaln_row_gather(rows, timestep_indices, token_tags, check_range=check_range)
        index = h3_adaln_indices(timestep_indices, token_tags)
        return tuple(_DeterministicRowGather.apply(rows, index).unbind(0))

    def forward_fp32(self, rows, timestep_indices, token_tags, *, check_range: bool = True):
        validate_h3_adaln_row_gather(rows, timestep_indices, token_tags, check_range=check_range)
        index = h3_adaln_indices(timestep_indices, token_tags)
        return tuple(
            chunk.index_select(0, index).float()
            for chunk in rows.double().chunk(H3_ADALN_CHUNKS, dim=-1)
        )
