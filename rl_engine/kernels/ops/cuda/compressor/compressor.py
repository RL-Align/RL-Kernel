# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 KV compressor: indexer entries (head 128) and main KV (head 512).

Thin wrapper around ``csrc/cuda/compressor/compress_fp8.cu``, byte-equal to the
goldens ``rl_engine.kernels.ops.pytorch.compressor.compress_indexer`` and
``compress_main``. The
state save (``save_states``) is an exact copy and runs as PyTorch indexing.
Fails closed when the compiled extension lacks the symbol.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

_SYMBOLS = ("compress_indexer_fp8", "compress_main_ds_mla")
FP8_DS_MLA_ROW = 576  # 448 FP8 + 64 BF16 per entry
FP8_DS_MLA_SCALES = 8  # 7 UE8M0 bytes + a zero pad per entry


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("compress_indexer requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    missing = [name for name in _SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def compress_indexer_cuda(
    state_cache: Tensor,
    block_table: Tensor,
    token_to_req: Tensor,
    positions: Tensor,
    rms_weight: Tensor,
    eps: float,
    cos_sin_cache: Tensor,
    k_cache: Tensor,
    k_block_size: int,
    k_slots: Tensor,
    ratio: int = 4,
    overlap: bool = True,
) -> None:
    """Write the FP8 indexer entry of every boundary token into ``k_cache``.

    ``k_cache`` is uint8 ``[blocks, >= k_block_size * 132]`` (``k_block_size``
    FP8 rows of 128, then their FP32 scales: DeepGEMM's paged layout);
    ``k_slots[t]`` is token ``t``'s entry slot, ``-1`` to skip.
    """
    _extension().compress_indexer_fp8(
        state_cache,
        block_table,
        token_to_req,
        positions,
        rms_weight,
        float(eps),
        cos_sin_cache,
        k_cache.view(k_cache.shape[0], -1),
        int(k_block_size),
        k_slots,
        int(ratio),
        bool(overlap),
    )


def compress_main_cuda(
    state_cache: Tensor,
    block_table: Tensor,
    token_to_req: Tensor,
    positions: Tensor,
    rms_weight: Tensor,
    eps: float,
    cos_sin_cache: Tensor,
    k_cache: Tensor,
    k_block_size: int,
    k_slots: Tensor,
    ratio: int,
) -> None:
    """Write the fp8_ds_mla entry of every boundary token into the main KV cache.

    ``ratio`` 4 (window 8 with overlap) or 128 (window 128). ``k_cache`` is
    uint8 ``[blocks, >= k_block_size * 584]``: ``k_block_size`` rows of 576
    bytes (448 FP8 + 64 BF16), then 8 scale bytes per entry.
    """
    _extension().compress_main_ds_mla(
        state_cache,
        block_table,
        token_to_req,
        positions,
        rms_weight,
        float(eps),
        cos_sin_cache,
        k_cache.view(k_cache.shape[0], -1),
        int(k_block_size),
        k_slots,
        int(ratio),
    )


def save_states_cuda(
    kv: Tensor, score: Tensor, ape: Tensor, positions: Tensor, state_cache: Tensor, slots: Tensor
) -> None:
    """CUDA ``save_states``: ``state[slot] = [kv | score + ape[pos % ratio]]``, skips ``slot < 0``.

    Equal to the golden ``rl_engine.kernels.ops.pytorch.compressor.save_states``
    (one FP32 add), without its host-side masking.
    """
    importlib.import_module("rl_engine._C").dsv4_save_states(
        kv,
        score,
        ape.contiguous(),
        positions.long().contiguous(),
        state_cache,
        slots.long().contiguous(),
    )
