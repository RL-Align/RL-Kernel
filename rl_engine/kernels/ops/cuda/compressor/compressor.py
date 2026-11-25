# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 KV compressor: ``csrc/cuda/compressor/compress_fp8.cu``"""

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
    importlib.import_module("rl_engine._C").dsv4_save_states(
        kv,
        score,
        ape.contiguous(),
        positions.long().contiguous(),
        state_cache,
        slots.long().contiguous(),
    )


def _ext41():
    ext = _extension()
    for name in ("dsv41_compress_main", "dsv41_store_rows"):
        if not hasattr(ext, name):
            raise RuntimeError(f"rl_engine._C lacks {name}; rebuild the CUDA extension")
    return ext


def dsv41_compress_main_cuda(
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
    latent: Tensor | None = None,
) -> None:
    """DSv4.1 main KV at ratio 2 (window 2, states saved with a zero APE) into the
    fp8_ds_mla cache; ``latent`` (BF16 [tokens, 512]) receives each entry's normed row
    before RoPE at its token row."""
    _ext41().dsv41_compress_main(
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
        2,
        latent,
    )


def dsv41_store_rows_cuda(
    rows: Tensor,
    positions: Tensor,
    rms_weight: Tensor,
    eps: float,
    cos_sin_cache: Tensor,
    k_cache: Tensor,
    k_block_size: int,
    k_slots: Tensor,
    ratio: int,
    latent: Tensor | None = None,
) -> None:
    """DSv4.1 entries without pooling from FP32 ``rows`` [tokens, D]: D = 512 the ratio-1
    main KV (fp8_ds_mla), D = 128 the indexer K ``wk(latent)`` (FP8 indexer entry)."""
    _ext41().dsv41_store_rows(
        rows,
        positions,
        rms_weight,
        float(eps),
        cos_sin_cache,
        k_cache.view(k_cache.shape[0], -1),
        int(k_block_size),
        k_slots,
        int(ratio),
        latent,
    )
