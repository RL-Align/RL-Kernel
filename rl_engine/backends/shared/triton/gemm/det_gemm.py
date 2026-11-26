# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch- and TP-invariant deterministic GEMM, Triton path.

BF16 outputs use the same arithmetic graph as the native deterministic kernel:
each canonical K-tree leaf contains at most 32 values, accumulates in FP32, and
rounds to BF16. Fixed subtrees can run in parallel, with BF16 rounding at
each midpoint merge. The strict ROCm leaf keeps the native gfx942 scalar FMA
order so both implementations have zero mismatch; a future MFMA leaf requires
a separately versioned arithmetic contract because its internal accumulation
does not match the scalar reference at every BF16 rounding boundary.

Tile sizes are pinned. FP32-output calls preserve the earlier fixed-order,
no-split-K contract and do not use BF16 tree nodes.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.utils.logger import logger

# Pinned. NOT autotuned (autotune picks per-shape configs -> breaks invariance).
_BLOCK_M, _BLOCK_N, _BLOCK_K = 64, 64, 32
_FIXED_PROJECTION_REDUCTIONS = (256, 4096, 6656, 19968)
# Keep the intermediate tree workspace bounded.  A VLLM warmup can present a
# very large flattened token dimension; allocating ``node_count * M * N`` in
# one shot would otherwise create a multi-terabyte virtual tensor on ROCm.
# Chunking over M preserves the exact K-tree and BF16 rounding contract.
_MAX_TREE_WORKSPACE_ELEMENTS = 128 * 1024 * 1024
_TREE_PLANS: dict[tuple[int, int, bool], "_DeviceTreePlan"] = {}
_TREE_PLAN_LOCK = threading.Lock()
_ROCM_TUNE_DECODE_LEAF = True
_ROCM_FUSE_LEAF_REDUCTION = True


@dataclass(frozen=True)
class _TreeLeafConfig:
    block_m: int
    block_n: int
    num_warps: int
    n_fastest: bool


_DEFAULT_TREE_LEAF_CONFIG = _TreeLeafConfig(_BLOCK_M, _BLOCK_N, 4, False)

# Offline-swept on ROCm gfx942 with Triton 3.7. These entries deliberately
# cover only the Qwen3-8B TP1 logical shapes and stride modes used by the FFN.
# Other architectures and shapes retain the established 64x64 specialization.
_GFX942_QWEN_FORWARD_LEAF_CONFIGS = {
    1: _TreeLeafConfig(1, 128, 1, True),
    8: _TreeLeafConfig(8, 128, 2, True),
    32: _TreeLeafConfig(32, 128, 4, True),
}
_GFX942_QWEN_WGRAD_LEAF_CONFIGS = {
    1: _TreeLeafConfig(128, 64, 2, True),
    8: _TreeLeafConfig(128, 64, 2, True),
    32: _TreeLeafConfig(64, 64, 2, True),
}
_QWEN_FORWARD_GEMM_SHAPES = {(4096, 12288), (12288, 4096)}
_QWEN_WGRAD_OUTPUT_SHAPES = {(4096, 12288), (12288, 4096)}

# Qwen3-8B TP2 local shards. TP4/8 candidates improved isolated leaves but did
# not clear the distributed end-to-end promotion threshold, so they deliberately
# retain the fallback. These entries were swept independently from the TP1 table
# because smaller local N/K changes both grid occupancy and the
# point where tree-reduction launch cost dominates leaf time. Keep the key
# exact: nearby shapes and non-target token counts retain the established
# fallback instead of inheriting a configuration from a different TP graph.
_GFX942_QWEN_TP_SHARD_FORWARD_LEAF_CONFIGS = {
    (16, 4096, 6144): _TreeLeafConfig(8, 64, 1, True),
    (16, 6144, 4096): _TreeLeafConfig(8, 64, 1, True),
    (32, 4096, 6144): _TreeLeafConfig(32, 128, 4, True),
    (32, 6144, 4096): _TreeLeafConfig(32, 128, 4, True),
}
_GFX942_QWEN_TP_SHARD_WGRAD_LEAF_CONFIGS = {
    (4096, 32, 6144): _TreeLeafConfig(128, 64, 2, True),
    (6144, 32, 4096): _TreeLeafConfig(128, 64, 2, True),
}

_GFX942_QWEN_TP4_DECODE_SHAPES = {
    (4096, 1536),  # QKV projection
    (1024, 4096),  # attention output projection
    (4096, 6144),  # gate/up projection
    (3072, 4096),  # down projection
    (4096, 37984),  # vocabulary-parallel LM head
}


def _gfx942_qwen_tp4_decode_leaf_config(
    m_size: int,
    k_size: int,
    n_size: int,
) -> _TreeLeafConfig | None:
    """Return the gfx942 small-batch schedule for exact Qwen3-8B TP4 shapes."""

    if not 1 <= m_size <= 32 or (k_size, n_size) not in _GFX942_QWEN_TP4_DECODE_SHAPES:
        return None
    block_m = 1 << (m_size - 1).bit_length()
    block_n = 64 if m_size == 1 or (k_size, n_size) == (3072, 4096) else 128
    num_warps = 1 if block_m <= 2 else 2 if block_m <= 8 else 4
    return _TreeLeafConfig(block_m, block_n, num_warps, True)


def _gfx942_qwen_tree_leaf_config(
    m_size: int,
    k_size: int,
    n_size: int,
    *,
    transpose_output: bool,
    preserve_a_strides: bool,
) -> _TreeLeafConfig:
    logical_shape = (m_size, k_size, n_size)
    if (
        not transpose_output
        and not preserve_a_strides
        and (k_size, n_size) in _QWEN_FORWARD_GEMM_SHAPES
    ):
        return _GFX942_QWEN_FORWARD_LEAF_CONFIGS.get(
            m_size,
            _DEFAULT_TREE_LEAF_CONFIG,
        )
    if not transpose_output and not preserve_a_strides:
        return _GFX942_QWEN_TP_SHARD_FORWARD_LEAF_CONFIGS.get(
            logical_shape,
            _DEFAULT_TREE_LEAF_CONFIG,
        )
    if transpose_output and preserve_a_strides and (m_size, n_size) in _QWEN_WGRAD_OUTPUT_SHAPES:
        return _GFX942_QWEN_WGRAD_LEAF_CONFIGS.get(
            k_size,
            _DEFAULT_TREE_LEAF_CONFIG,
        )
    if transpose_output and preserve_a_strides:
        return _GFX942_QWEN_TP_SHARD_WGRAD_LEAF_CONFIGS.get(
            logical_shape,
            _DEFAULT_TREE_LEAF_CONFIG,
        )
    return _DEFAULT_TREE_LEAF_CONFIG


@functools.lru_cache(maxsize=None)
def _device_arch(device_index: int) -> str:
    if getattr(torch.version, "hip", None) is None:
        return ""
    properties = torch.cuda.get_device_properties(device_index)
    return str(getattr(properties, "gcnArchName", "")).partition(":")[0]


@functools.lru_cache(maxsize=None)
def _tree_leaf_config(
    device: torch.device,
    m_size: int,
    k_size: int,
    n_size: int,
    *,
    transpose_output: bool,
    preserve_a_strides: bool,
) -> _TreeLeafConfig:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if _device_arch(device_index) != "gfx942":
        return _DEFAULT_TREE_LEAF_CONFIG
    return _gfx942_qwen_tree_leaf_config(
        m_size,
        k_size,
        n_size,
        transpose_output=transpose_output,
        preserve_a_strides=preserve_a_strides,
    )


@dataclass(frozen=True)
class _TreePlan:
    leaf_starts: tuple[int, ...]
    leaf_lengths: tuple[int, ...]
    leaf_nodes: tuple[int, ...]
    reduction_levels: tuple[tuple[tuple[int, int, int], ...], ...]
    root: int
    node_count: int


@dataclass(frozen=True)
class _DeviceTreePlan:
    host: _TreePlan
    leaf_starts: torch.Tensor
    leaf_lengths: torch.Tensor
    leaf_nodes: torch.Tensor
    reduction_levels: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    rocm_fused_reduction_pairs: tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...
    ]
    rocm_leaf_reduction: (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    )
    rocm_fused_reduction_pairs_after_leaf: tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...
    ]


def _build_tree_plan(k_size: int, *, fixed_leaf_tree: bool = False) -> _TreePlan:
    if k_size <= 0:
        raise ValueError(f"deterministic GEMM K must be positive, got {k_size}")

    leaf_starts: list[int] = []
    leaf_lengths: list[int] = []
    leaf_nodes: list[int] = []
    reductions_by_height: dict[int, list[tuple[int, int, int]]] = {}
    next_node = 0

    def visit(begin: int, end: int) -> tuple[int, int]:
        nonlocal next_node
        if end - begin <= _BLOCK_K:
            node = next_node
            next_node += 1
            leaf_starts.append(begin)
            leaf_lengths.append(end - begin)
            leaf_nodes.append(node)
            return node, 0

        if fixed_leaf_tree:
            leaves = (end - begin + _BLOCK_K - 1) // _BLOCK_K
            midpoint = begin + (leaves // 2) * _BLOCK_K
        else:
            midpoint = begin + (end - begin) // 2
        lower, lower_height = visit(begin, midpoint)
        upper, upper_height = visit(midpoint, end)
        node = next_node
        next_node += 1
        height = max(lower_height, upper_height) + 1
        reductions_by_height.setdefault(height, []).append((lower, upper, node))
        return node, height

    root, max_height = visit(0, k_size)
    reduction_levels = tuple(
        tuple(reductions_by_height.get(height, ())) for height in range(1, max_height + 1)
    )
    return _TreePlan(
        leaf_starts=tuple(leaf_starts),
        leaf_lengths=tuple(leaf_lengths),
        leaf_nodes=tuple(leaf_nodes),
        reduction_levels=reduction_levels,
        root=root,
        node_count=next_node,
    )


def _device_tree_plan(
    k_size: int, device: torch.device, *, fixed_leaf_tree: bool = False
) -> _DeviceTreePlan:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device_index, k_size, fixed_leaf_tree)
    with _TREE_PLAN_LOCK:
        cached = _TREE_PLANS.get(key)
        if cached is not None:
            return cached

        host = _build_tree_plan(k_size, fixed_leaf_tree=fixed_leaf_tree)

        def indices(values: tuple[int, ...]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        levels = []
        for operations in host.reduction_levels:
            lower_nodes, upper_nodes, output_nodes = zip(*operations, strict=True)
            levels.append((indices(lower_nodes), indices(upper_nodes), indices(output_nodes)))
        fused_pairs = []
        leaf_reduction = None
        fused_pairs_after_leaf = []
        # Fused pairs assume a complete binary tree. Fixed 32-wide leaves
        # may have different heights, so run their explicit reduction levels.
        if not fixed_leaf_tree and _device_arch(device_index) == "gfx942":

            def build_fused_pairs(start_level: int):
                result = []
                for level_index in range(start_level, len(host.reduction_levels) - 1, 2):
                    first = {
                        output: (lower, upper)
                        for lower, upper, output in host.reduction_levels[level_index]
                    }
                    grandchildren = []
                    outputs = []
                    for lower, upper, output in host.reduction_levels[level_index + 1]:
                        grandchildren.append((*first[lower], *first[upper]))
                        outputs.append(output)
                    node0, node1, node2, node3 = zip(*grandchildren, strict=True)
                    result.append(
                        (
                            indices(node0),
                            indices(node1),
                            indices(node2),
                            indices(node3),
                            indices(tuple(outputs)),
                        )
                    )
                return result

            fused_pairs = build_fused_pairs(0)
            fused_pairs_after_leaf = build_fused_pairs(1)
            if host.reduction_levels:
                first_level = host.reduction_levels[0]
                paired_nodes = {node for op in first_level for node in op[:2]}
                if paired_nodes == set(host.leaf_nodes):
                    leaf_metadata = {
                        node: (start, length)
                        for start, length, node in zip(
                            host.leaf_starts,
                            host.leaf_lengths,
                            host.leaf_nodes,
                            strict=True,
                        )
                    }
                    lower_starts, lower_lengths = [], []
                    upper_starts, upper_lengths, outputs = [], [], []
                    for lower, upper, output in first_level:
                        lower_start, lower_length = leaf_metadata[lower]
                        upper_start, upper_length = leaf_metadata[upper]
                        lower_starts.append(lower_start)
                        lower_lengths.append(lower_length)
                        upper_starts.append(upper_start)
                        upper_lengths.append(upper_length)
                        outputs.append(output)
                    leaf_reduction = (
                        indices(tuple(lower_starts)),
                        indices(tuple(lower_lengths)),
                        indices(tuple(upper_starts)),
                        indices(tuple(upper_lengths)),
                        indices(tuple(outputs)),
                    )
        result = _DeviceTreePlan(
            host=host,
            leaf_starts=indices(host.leaf_starts),
            leaf_lengths=indices(host.leaf_lengths),
            leaf_nodes=indices(host.leaf_nodes),
            reduction_levels=tuple(levels),
            rocm_fused_reduction_pairs=tuple(fused_pairs),
            rocm_leaf_reduction=leaf_reduction,
            rocm_fused_reduction_pairs_after_leaf=tuple(fused_pairs_after_leaf),
        )
        _TREE_PLANS[key] = result
        return result


if _TRITON_AVAILABLE:

    @triton.jit
    def _fixed_add(left, right):
        return (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16)

    @triton.jit
    def _fixed_ordered_fmas(
        x, w, acc, WIDTH: tl.constexpr, OUTER: tl.constexpr, COLUMN_FIRST: tl.constexpr = False
    ):
        # Load a whole K slice, then consume its registers in ascending order.
        # Splitting into contiguous halves gives the compiler a static index
        # for every operand without changing the FP32 dependency chain.
        if WIDTH == 1:
            a = tl.reshape(x, (x.shape[0],))
            b = tl.reshape(w, (w.shape[0],))
            if COLUMN_FIRST:
                return tl.fma(a[None, :], b[:, None], acc)
            elif OUTER:
                return tl.fma(a[:, None], b[None, :], acc)
            else:
                return tl.fma(a, b, acc)
        else:
            xl, xr = tl.split(x.reshape(x.shape[0], 2, WIDTH // 2).permute(0, 2, 1))
            wl, wr = tl.split(w.reshape(w.shape[0], 2, WIDTH // 2).permute(0, 2, 1))
            acc = _fixed_ordered_fmas(xl, wl, acc, WIDTH // 2, OUTER, COLUMN_FIRST)
            return _fixed_ordered_fmas(xr, wr, acc, WIDTH // 2, OUTER, COLUMN_FIRST)

    @triton.jit
    def _fixed_subtree(group, leaves, LEVELS: tl.constexpr):
        start = group * 0
        count = start + leaves
        for level in tl.static_range(LEVELS):
            right = (group & (1 << (LEVELS - level - 1))) != 0
            half = count // 2
            start += tl.where(right, half, 0)
            count = tl.where(right, count - half, half)
        return start, count

    @triton.jit
    def _fixed_root(nodes, BLOCK: tl.constexpr, LEVELS: tl.constexpr):
        for level in tl.static_range(LEVELS):
            left, right = tl.split(tl.reshape(nodes, (BLOCK, 1 << (LEVELS - level - 1), 2)))
            nodes = _fixed_add(left, right)
        return tl.reshape(nodes, (BLOCK,))

    @triton.jit
    def _fixed_decode_kernel(
        a, b, dst, BN: tl.constexpr, LEAVES: tl.constexpr, LEVELS: tl.constexpr
    ):
        col = tl.program_id(0).to(tl.int64)
        width: tl.constexpr = 1 << LEVELS
        padded: tl.constexpr = width if LEAVES == width else 2 * width
        leaf = tl.arange(0, padded)
        acc = tl.full((padded,), 0, tl.float32)
        for offset in tl.static_range(0, 32, 8):
            k = tl.minimum(leaf, LEAVES - 1)[:, None] * 32 + offset + tl.arange(0, 8)[None, :]
            x = tl.load(a + k).to(tl.float32)
            w = tl.load(b + col * BN + k).to(tl.float32)
            acc = _fixed_ordered_fmas(x, w, acc, 8, False)
        nodes = acc.to(tl.bfloat16)
        if LEAVES != width:
            start, count = _fixed_subtree(tl.arange(0, width), LEAVES, LEVELS)
            left = tl.gather(nodes, start, axis=0)
            right = tl.gather(nodes, start + 1, axis=0)
            nodes = tl.where(count == 2, _fixed_add(left, right), left)
        root = _fixed_root(nodes, 1, LEVELS)
        tl.store(dst + col + tl.arange(0, 1), root)

    @triton.jit
    def _fixed_full_leaf(
        a,
        b,
        rows,
        cols,
        start,
        M,
        N: tl.constexpr,
        AM: tl.constexpr,
        AK: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
        PACK: tl.constexpr = 1,
        COLUMN_FIRST: tl.constexpr = False,
    ):
        if COLUMN_FIRST:
            acc = tl.full((cols.shape[0], rows.shape[0]), 0, tl.float32)
        else:
            acc = tl.full((rows.shape[0], cols.shape[0]), 0, tl.float32)
        for offset in tl.static_range(0, 32, PACK):
            k = (start + offset + tl.arange(0, PACK)).to(tl.int64)
            x = tl.load(a + rows[:, None] * AM + k[None, :] * AK, rows[:, None] < M, 0)
            w = tl.load(b + cols[:, None] * BN + k[None, :] * BK, cols[:, None] < N, 0)
            acc = _fixed_ordered_fmas(
                x.to(tl.float32), w.to(tl.float32), acc, PACK, True, COLUMN_FIRST
            )
        return acc.to(tl.bfloat16)

    @triton.jit
    def _fixed_partial_wgrad_leaf(
        a,
        b,
        rows,
        cols,
        start,
        length,
        M,
        N: tl.constexpr,
        AM: tl.constexpr,
        AK: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
    ):
        acc = tl.full((rows.shape[0], cols.shape[0]), 0, tl.float32)
        for pack in range(0, length, 4):
            for offset in tl.static_range(4):
                active = pack + offset < length
                k = (start + pack + offset).to(tl.int64)
                x = tl.load(a + rows * AM + k * AK, (rows < M) & active, 0).to(tl.float32)
                w = tl.load(b + cols * BN + k * BK, (cols < N) & active, 0).to(tl.float32)
                # Inactive FMAs must preserve the sign of a zero accumulator.
                acc = tl.where(active, tl.fma(x[:, None], w[None, :], acc), acc)
        return acc.to(tl.bfloat16)

    @triton.jit
    def _fixed_tile_node(
        a,
        b,
        rows,
        cols,
        start,
        count,
        M,
        N: tl.constexpr,
        AM: tl.constexpr,
        AK: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
        PACK: tl.constexpr,
        COLUMN_FIRST: tl.constexpr = False,
        tail_length=0,
    ):
        # Each half of this two- to four-leaf subtree contains one or two leaves.
        half = count // 2
        left = _fixed_full_leaf(
            a, b, rows, cols, start * 32, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
        )
        if half == 2:
            sibling = _fixed_full_leaf(
                a, b, rows, cols, (start + 1) * 32, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
            )
            left = _fixed_add(left, sibling)
        right_start = (start + half) * 32
        if count - half == 1 and tail_length != 0:
            right = _fixed_partial_wgrad_leaf(
                a, b, rows, cols, right_start, tail_length, M, N, AM, AK, BK, BN
            )
        else:
            right = _fixed_full_leaf(
                a, b, rows, cols, right_start, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
            )
            if count - half == 2:
                if tail_length != 0:
                    sibling = _fixed_partial_wgrad_leaf(
                        a, b, rows, cols, right_start + 32, tail_length, M, N, AM, AK, BK, BN
                    )
                else:
                    sibling = _fixed_full_leaf(
                        a, b, rows, cols, right_start + 32, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
                    )
                right = _fixed_add(right, sibling)
        return _fixed_add(left, right)

    @triton.jit(do_not_specialize=["M"])
    def _fixed_projection_leaves_kernel(
        a,
        b,
        dst,
        M,
        N: tl.constexpr,
        AM: tl.constexpr,
        AK: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        WGRAD_96: tl.constexpr,
        LEAVES: tl.constexpr = 0,
        GROUP_LEVELS: tl.constexpr = 0,
        PACK: tl.constexpr = 1,
        COLUMN_FIRST: tl.constexpr = False,
        STATIC_M: tl.constexpr = 0,
    ):
        if STATIC_M:
            M = STATIC_M
        if WGRAD_96:
            # Visit consecutive tiles along the contiguous feature axis of dW.
            n_tiles: tl.constexpr = triton.cdiv(N, BLOCK_N)
            pid = tl.program_id(0)
            row_tile, col_tile = (pid // n_tiles).to(tl.int64), (pid % n_tiles).to(tl.int64)
        else:
            row_tile, col_tile = tl.program_id(1), tl.program_id(0)
        rows = (row_tile * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        cols = (col_tile * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        group = tl.program_id(2).to(tl.int64)
        if WGRAD_96:
            value = _fixed_tile_node(
                a, b, rows, cols, 0, 3, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
            )
        elif LEAVES == 8 and GROUP_LEVELS == 0:
            left = _fixed_tile_node(
                a, b, rows, cols, 0, 4, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
            )
            right = _fixed_tile_node(
                a, b, rows, cols, 4, 4, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
            )
            value = _fixed_add(left, right)
        elif LEAVES:
            start, count = _fixed_subtree(group, LEAVES, GROUP_LEVELS)
            value = _fixed_tile_node(
                a, b, rows, cols, start, count, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
            )
        else:
            value = _fixed_full_leaf(
                a, b, rows, cols, group * 32, M, N, AM, AK, BK, BN, PACK, COLUMN_FIRST
            )
        if COLUMN_FIRST:
            row, col = rows[None, :], cols[:, None]
        else:
            row, col = rows[:, None], cols[None, :]
        offsets = (group * M + row) * N + col
        tl.store(dst + offsets, value, mask=(row < M) & (col < N))

    @triton.jit
    def _fixed_wgrad_small_tree(
        a,
        b,
        rows,
        cols,
        length,
        M,
        N: tl.constexpr,
        AM: tl.constexpr,
        AK: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
        START: tl.constexpr,
        LEAVES: tl.constexpr,
        HAS_TAIL: tl.constexpr,
    ):
        if LEAVES == 1:
            if HAS_TAIL:
                return _fixed_partial_wgrad_leaf(
                    a, b, rows, cols, START * 32, length, M, N, AM, AK, BK, BN
                )
            return _fixed_full_leaf(a, b, rows, cols, START * 32, M, N, AM, AK, BK, BN, 8)
        else:
            half: tl.constexpr = LEAVES // 2
            left = _fixed_wgrad_small_tree(
                a, b, rows, cols, length, M, N, AM, AK, BK, BN, START, half, False
            )
            right = _fixed_wgrad_small_tree(
                a,
                b,
                rows,
                cols,
                length,
                M,
                N,
                AM,
                AK,
                BK,
                BN,
                START + half,
                LEAVES - half,
                HAS_TAIL,
            )
            return _fixed_add(left, right)

    @triton.jit(do_not_specialize=["tokens"])
    def _fixed_wgrad_kernel(
        a,
        b,
        dst,
        M,
        tokens,
        N: tl.constexpr,
        AM: tl.constexpr,
        AK: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        DIRECT_LEAVES: tl.constexpr,
        GROUP_LEVELS: tl.constexpr,
        HAS_TAIL: tl.constexpr,
        GROUP_LEAVES: tl.constexpr = 0,
    ):
        n_tiles: tl.constexpr = triton.cdiv(N, BLOCK_N)
        pid = tl.program_id(0)
        row_tile, col_tile = pid // n_tiles, pid % n_tiles
        if not DIRECT_LEAVES:
            if GROUP_LEVELS <= 15:
                group = tl.program_id(1)
            else:
                group = row_tile // tl.cdiv(M, BLOCK_M)
                row_tile %= tl.cdiv(M, BLOCK_M)
        rows = row_tile.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = col_tile.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
        if DIRECT_LEAVES:
            value = _fixed_wgrad_small_tree(
                a, b, rows, cols, tokens % 32, M, N, AM, AK, BK, BN, 0, DIRECT_LEAVES, HAS_TAIL
            )
        else:
            if GROUP_LEAVES:
                start, count = group * GROUP_LEAVES, GROUP_LEAVES
            else:
                leaves = 1 + (tokens - 1) // 32
                start, count = _fixed_subtree(group, leaves, GROUP_LEVELS)
            if HAS_TAIL:
                length = tl.where(group == (1 << GROUP_LEVELS) - 1, tokens % 32, 0)
            else:
                length = 0
            value = _fixed_tile_node(
                a, b, rows, cols, start, count, M, N, AM, AK, BK, BN, 8, tail_length=length
            )
        offsets = rows[:, None] * N + cols[None, :]
        if not DIRECT_LEAVES:
            offsets += group.to(tl.int64) * M * N
        tl.store(dst + offsets, value, (rows[:, None] < M) & (cols[None, :] < N))

    @triton.jit
    def _fixed_wgrad_large_root(
        leaves_ptr, offsets, elements, BLOCK: tl.constexpr, LEVELS: tl.constexpr
    ):
        # Keep per-CTA storage bounded even when there are thousands of
        # subtree roots. Each chunk is itself a complete canonical subtree.
        chunk_levels: tl.constexpr = 7
        chunk_groups: tl.constexpr = 1 << chunk_levels
        chunks: tl.constexpr = 1 << (LEVELS - chunk_levels)
        depth_slots: tl.constexpr = triton.next_power_of_2(LEVELS - chunk_levels)
        slots = tl.arange(0, depth_slots)
        pending = tl.full((BLOCK, depth_slots), 0, tl.bfloat16)
        root = tl.full((BLOCK,), 0, tl.bfloat16)
        for chunk in range(chunks):
            groups = chunk * chunk_groups + tl.arange(0, chunk_groups)
            nodes = tl.load(
                leaves_ptr + groups[None, :] * elements + offsets[:, None],
                offsets[:, None] < elements,
                0,
            )
            root = _fixed_root(nodes, BLOCK, chunk_levels)
            path = chunk
            depth = 0
            while (path & 1) != 0:
                index = tl.full((BLOCK, 1), depth, tl.int32)
                left = tl.gather(pending, index, axis=1).reshape((BLOCK,))
                root = _fixed_add(left, root)
                path //= 2
                depth += 1
            pending = tl.where(slots[None, :] == depth, root[:, None], pending)
        return root

    @triton.jit(do_not_specialize=["M"])
    def _fixed_projection_root_kernel(
        leaves_ptr,
        output_ptr,
        M,
        N: tl.constexpr,
        OUTPUT_STRIDE: tl.constexpr,
        LEAVES: tl.constexpr,
        LEVELS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        width: tl.constexpr = 1 << LEVELS
        elements = M * N
        if LEAVES == width and width > 4096:
            root = _fixed_wgrad_large_root(leaves_ptr, offsets, elements, BLOCK, LEVELS)
        elif LEAVES == width:
            group = tl.arange(0, width)
            nodes = tl.load(
                leaves_ptr + group[None, :] * elements + offsets[:, None],
                mask=offsets[:, None] < elements,
                other=0,
            )
            root = _fixed_root(nodes, BLOCK, LEVELS)
        else:
            # At this depth every midpoint-tree branch has one or two leaves.
            # Descend with the same floor midpoint as the registered reference.
            start, count = _fixed_subtree(tl.arange(0, width), LEAVES, LEVELS)
            ptr = leaves_ptr + start[None, :] * elements + offsets[:, None]
            lower = tl.load(ptr, mask=offsets[:, None] < elements, other=0)
            paired = count[None, :] == 2
            upper = tl.load(ptr + elements, mask=(offsets[:, None] < elements) & paired, other=0)
            nodes = tl.where(paired, _fixed_add(lower, upper), lower)
            root = _fixed_root(nodes, BLOCK, LEVELS)

        # These levels are complete. Merge adjacent children and round every
        # parent, including the root, before writing the final output.
        output_offsets = (offsets // N).to(tl.int64) * OUTPUT_STRIDE + offsets % N
        tl.store(output_ptr + output_offsets, root, mask=offsets < elements)

    @triton.jit(do_not_specialize=["M"])
    def _det_gemm_tree_leaf_kernel(
        a_ptr,
        b_ptr,
        workspace_ptr,
        leaf_starts_ptr,
        leaf_lengths_ptr,
        leaf_nodes_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        N_FASTEST: tl.constexpr,
        FIXED_LEAVES: tl.constexpr = False,
    ):
        if N_FASTEST:
            pid_n = tl.program_id(0)
            pid_m = tl.program_id(1)
            leaf = tl.program_id(2)
        else:
            leaf = tl.program_id(0)
            pid_m = tl.program_id(1)
            pid_n = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # Qwen prefill workspaces can exceed 2**31 elements.  ROCm Triton
        # otherwise keeps the index expression in i32 and wraps addresses,
        # causing GPU memory faults for large M*N trees.
        leaf_start = tl.load(leaf_starts_ptr + leaf).to(tl.int64)
        leaf_length = tl.load(leaf_lengths_ptr + leaf).to(tl.int64)
        leaf_node = tl.load(leaf_nodes_ptr + leaf).to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Keep the leaf's ascending scalar FMA order identical to the native
        # gfx942 correctness kernel.  A tl.dot/MFMA leaf is topology-stable but
        # differs from the scalar reference at rare BF16 rounding boundaries.
        for offset in tl.static_range(0, BLOCK_K):
            k_offset = leaf_start + offset
            active = offset < leaf_length
            a = tl.load(
                a_ptr + offs_m * stride_am + k_offset * stride_ak,
                mask=(offs_m < M) & active,
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + k_offset * stride_bk + offs_n * stride_bn,
                mask=(offs_n < N) & active,
                other=0.0,
            ).to(tl.float32)
            if FIXED_LEAVES:
                # Padded offsets in a short leaf must preserve the accumulator,
                # including the sign of a zero produced by the last real FMA.
                acc = tl.where(active, tl.fma(a[:, None], b[None, :], acc), acc)
            else:
                acc += a[:, None] * b[None, :]
        output_offsets = (
            leaf_node * (M * N) + offs_m[:, None].to(tl.int64) * N + offs_n[None, :].to(tl.int64)
        )
        output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            workspace_ptr + output_offsets,
            acc.to(workspace_ptr.dtype.element_ty),
            mask=output_mask,
        )

    # Preserve the WS1 manifest's historical kernel symbol while retaining the
    # more descriptive implementation name used by the optimized tree path.
    _det_gemm_kernel = _det_gemm_tree_leaf_kernel

    @triton.jit(do_not_specialize=["M"])
    def _det_gemm_tree_leaf_reduce_rocm_kernel(
        a_ptr,
        b_ptr,
        workspace_ptr,
        output_ptr,
        lower_starts_ptr,
        lower_lengths_ptr,
        upper_starts_ptr,
        upper_lengths_ptr,
        output_nodes_ptr,
        M,
        N: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        N_FASTEST: tl.constexpr,
        WRITE_OUTPUT: tl.constexpr,
    ):
        if N_FASTEST:
            pid_n = tl.program_id(0)
            pid_m = tl.program_id(1)
            pair = tl.program_id(2)
        else:
            pair = tl.program_id(0)
            pid_m = tl.program_id(1)
            pid_n = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        lower_start = tl.load(lower_starts_ptr + pair).to(tl.int64)
        lower_length = tl.load(lower_lengths_ptr + pair).to(tl.int64)
        upper_start = tl.load(upper_starts_ptr + pair).to(tl.int64)
        upper_length = tl.load(upper_lengths_ptr + pair).to(tl.int64)

        lower_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for offset in tl.static_range(0, BLOCK_K):
            k_offset = lower_start + offset
            active = offset < lower_length
            a = tl.load(
                a_ptr + offs_m * stride_am + k_offset * stride_ak,
                mask=(offs_m < M) & active,
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + k_offset * stride_bk + offs_n * stride_bn,
                mask=(offs_n < N) & active,
                other=0.0,
            ).to(tl.float32)
            lower_acc += a[:, None] * b[None, :]
        # This conversion is the original lower-leaf workspace store boundary.
        lower = lower_acc.to(workspace_ptr.dtype.element_ty)

        upper_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for offset in tl.static_range(0, BLOCK_K):
            k_offset = upper_start + offset
            active = offset < upper_length
            a = tl.load(
                a_ptr + offs_m * stride_am + k_offset * stride_ak,
                mask=(offs_m < M) & active,
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + k_offset * stride_bk + offs_n * stride_bn,
                mask=(offs_n < N) & active,
                other=0.0,
            ).to(tl.float32)
            upper_acc += a[:, None] * b[None, :]
        # Match both leaf BF16 stores followed by the original first-level FP32
        # add and BF16 store. Only the intermediate global-memory trip is gone.
        result = lower.to(tl.float32) + upper_acc.to(workspace_ptr.dtype.element_ty).to(tl.float32)
        output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        offsets = offs_m[:, None].to(tl.int64) * N + offs_n[None, :].to(tl.int64)
        if WRITE_OUTPUT:
            tl.store(
                output_ptr + offsets,
                result.to(output_ptr.dtype.element_ty),
                mask=output_mask,
            )
        else:
            output_node = tl.load(output_nodes_ptr + pair).to(tl.int64)
            tl.store(
                workspace_ptr + output_node * (M * N) + offsets,
                result.to(workspace_ptr.dtype.element_ty),
                mask=output_mask,
            )

    @triton.jit(do_not_specialize=["M"])
    def _det_gemm_tree_reduce_kernel(
        workspace_ptr,
        lower_nodes_ptr,
        upper_nodes_ptr,
        output_nodes_ptr,
        M,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCKS_ON_X: tl.constexpr = False,
    ):
        if BLOCKS_ON_X:
            block = tl.program_id(0)
            operation = tl.program_id(1)
        else:
            operation = tl.program_id(0)
            block = tl.program_id(1)
        offsets = (block * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        elements = M * N
        mask = offsets < elements
        lower_node = tl.load(lower_nodes_ptr + operation).to(tl.int64)
        upper_node = tl.load(upper_nodes_ptr + operation).to(tl.int64)
        output_node = tl.load(output_nodes_ptr + operation).to(tl.int64)
        lower = tl.load(
            workspace_ptr + lower_node * elements + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        upper = tl.load(
            workspace_ptr + upper_node * elements + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        result = lower + upper
        tl.store(
            workspace_ptr + output_node * elements + offsets,
            result.to(workspace_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _det_gemm_tree_reduce_to_output_rocm_kernel(
        workspace_ptr,
        output_ptr,
        lower_nodes_ptr,
        upper_nodes_ptr,
        # M is runtime-sized so decode batch changes reuse one compiled
        # reduction kernel instead of triggering a Triton JIT per batch size.
        # The reduction order and BF16 store boundary are unchanged.
        M,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        elements = M * N
        mask = offsets < elements
        lower_node = tl.load(lower_nodes_ptr).to(tl.int64)
        upper_node = tl.load(upper_nodes_ptr).to(tl.int64)
        lower = tl.load(
            workspace_ptr + lower_node * elements + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        upper = tl.load(
            workspace_ptr + upper_node * elements + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        result = lower + upper
        # Preserve the canonical root's FP32 add and BF16 store boundary; only
        # its destination changes, avoiding a device-to-device copy launch.
        tl.store(
            output_ptr + offsets,
            result.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _det_gemm_tree_reduce_two_levels_rocm_kernel(
        workspace_ptr,
        output_ptr,
        node0_ptr,
        node1_ptr,
        node2_ptr,
        node3_ptr,
        output_nodes_ptr,
        M,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        WRITE_OUTPUT: tl.constexpr,
    ):
        operation = tl.program_id(0)
        block = tl.program_id(1)
        offsets = (block * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        elements = M * N
        mask = offsets < elements
        node0 = tl.load(node0_ptr + operation).to(tl.int64)
        node1 = tl.load(node1_ptr + operation).to(tl.int64)
        node2 = tl.load(node2_ptr + operation).to(tl.int64)
        node3 = tl.load(node3_ptr + operation).to(tl.int64)
        value0 = tl.load(workspace_ptr + node0 * elements + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        value1 = tl.load(workspace_ptr + node1 * elements + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        value2 = tl.load(workspace_ptr + node2 * elements + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        value3 = tl.load(workspace_ptr + node3 * elements + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        # Match the two original kernel boundaries exactly: each first-level
        # FP32 add is rounded to BF16 before the second-level FP32 add.
        lower = (value0 + value1).to(workspace_ptr.dtype.element_ty).to(tl.float32)
        upper = (value2 + value3).to(workspace_ptr.dtype.element_ty).to(tl.float32)
        result = lower + upper
        if WRITE_OUTPUT:
            tl.store(
                output_ptr + offsets,
                result.to(output_ptr.dtype.element_ty),
                mask=mask,
            )
        else:
            output_node = tl.load(output_nodes_ptr + operation).to(tl.int64)
            tl.store(
                workspace_ptr + output_node * elements + offsets,
                result.to(workspace_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.jit
    def _copy_tree_root_kernel(
        workspace_ptr,
        output_ptr,
        root,
        elements,
        BLOCK: tl.constexpr,
    ):
        offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        mask = offsets < elements
        values = tl.load(workspace_ptr + root * elements + offsets, mask=mask)
        tl.store(output_ptr + offsets, values, mask=mask)

    @triton.jit
    def _copy_tree_root_transposed_kernel(
        workspace_ptr,
        output_ptr,
        root,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets_m = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offsets_n = (tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        elements = M * N
        mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        values = tl.load(
            workspace_ptr + root * elements + offsets_m[:, None] * N + offsets_n[None, :],
            mask=mask,
        )
        # Store the already-rounded BF16 root directly in [N, M] layout.
        # This is a pure address permutation: the canonical GEMM leaves, tree,
        # operand order, and every rounding boundary remain unchanged.
        tl.store(
            output_ptr + offsets_n[:, None] * M + offsets_m[None, :],
            tl.trans(values),
            mask=tl.trans(mask),
        )

    @triton.jit
    def _det_gemm_fp32_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PROMOTE_INPUTS: tl.constexpr,
    ):
        # One program = one output tile, walks the whole K in fixed order.
        # No split-K -> K-accumulation order independent of M -> batch-invariant.
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            if PROMOTE_INPUTS:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = acc.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)


def _triton_gemm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.contiguous(), b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    _det_gemm_fp32_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        PROMOTE_INPUTS=True,
    )
    return c


def _triton_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compatibility entry point for canonical Triton linear operators."""

    if output_dtype is None or output_dtype == torch.bfloat16:
        return _triton_tree_gemm(a, b)
    if output_dtype == torch.float32:
        return _triton_gemm_fp32(a, b)
    raise TypeError(f"unsupported deterministic Triton GEMM output dtype: {output_dtype}")


def _triton_fixed_decode(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One token and native [N,K] weights with a contiguous reduction dimension."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")
    n, k = weight.shape
    leaves = k // _BLOCK_K
    result = x.new_empty((1, n))
    _fixed_decode_kernel[(n,)](
        x,
        weight,
        result,
        weight.stride(0),
        leaves,
        leaves.bit_length() - 1,
        num_warps=8 if leaves > 256 else 4,
    )
    return result


def _triton_fixed_projection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    if m == 1 and a.stride(1) == 1 and b.stride(0) == 1 and n <= 2**31 - 1:
        return _triton_fixed_decode(a, b.T)

    leaves = k // _BLOCK_K
    result = a.new_empty((m, n))
    grouped = m != 1
    if grouped and leaves == 8:
        # The K/V input gradient fits its entire eight-leaf tree in one tile.
        max_rows = 65535 * 16
        for row in range(0, m, max_rows):
            chunk_m = min(max_rows, m - row)
            grid = (triton.cdiv(n, 64), triton.cdiv(chunk_m, 16), 1)
            _fixed_projection_leaves_kernel[grid](
                a[row:] if row else a,
                b,
                result[row:] if row else result,
                chunk_m,
                n,
                *a.stride(),
                *b.stride(),
                BLOCK_M=16,
                BLOCK_N=64,
                WGRAD_96=False,
                LEAVES=8,
                PACK=8,
                num_warps=4,
            )
        return result
    if grouped:
        block_m, block_n = 32, 128
        groups = triton.next_power_of_2(leaves) // 4
        # The wide, non-power-of-two projections reuse a column-major register tile.
        # The 128-leaf O projection has a fully unrolled four-leaf subtree.
        column_first = b.stride(0) == 1 and k in (6656, 19968) and n >= 4096
        pack = 16 if b.stride(0) == 1 and not column_first else 8
    else:
        block_m, block_n = 1, 64
        groups, pack, column_first = leaves, 1, False

    # On Hopper, merging more adjacent FFN outputs per block speeds up workspace reads.
    root_block = 32
    if (
        column_first
        and (k, n) in ((6656, 19968), (19968, 6656))
        and torch.cuda.get_device_capability(a.device)[0] == 9
    ):
        root_block = 128

    # Size column chunks to fit the 96-row training schedule. Balance their
    # tile counts so a narrow final column chunk does not limit utilization.
    # Larger batches reuse the same scratch allocation across row chunks.
    max_outputs = _MAX_TREE_WORKSPACE_ELEMENTS // groups
    column_tiles = max(1, max_outputs // ((96 if grouped else 1) * block_n))
    tiles_n = triton.cdiv(n, block_n)
    column_chunks = triton.cdiv(tiles_n, column_tiles)
    columns = min(n, triton.cdiv(tiles_n, column_chunks) * block_n, max_outputs)
    row_limit = min(65535 * block_m, max_outputs // columns)
    # Whole tiles keep row starts aligned for vector loads and stores.
    if row_limit >= block_m:
        row_limit = row_limit // block_m * block_m
    rows = min(m, row_limit)
    workspace = a.new_empty((groups, rows, columns))
    levels = groups.bit_length() - 1
    # In the 96-token training case, the compiler can eliminate the row masks.
    static_m = 96 if m == rows == 96 else 0
    for row in range(0, m, rows):
        chunk_m = min(rows, m - row)
        lhs = a if row == 0 else a[row:]
        for col in range(0, n, columns):
            chunk_n = min(columns, n - col)
            grid = (triton.cdiv(chunk_n, block_n), triton.cdiv(chunk_m, block_m), groups)
            _fixed_projection_leaves_kernel[grid](
                lhs,
                b if col == 0 else b[:, col:],
                workspace,
                chunk_m,
                chunk_n,
                *a.stride(),
                *b.stride(),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                WGRAD_96=False,
                LEAVES=leaves if grouped else 0,
                GROUP_LEVELS=levels,
                PACK=pack,
                COLUMN_FIRST=column_first,
                STATIC_M=static_m,
                num_warps=4 if grouped else 2,
            )
            _fixed_projection_root_kernel[(triton.cdiv(chunk_m * chunk_n, root_block),)](
                workspace,
                result[row:, col:] if row or col else result,
                chunk_m,
                chunk_n,
                n,
                groups,
                levels,
                BLOCK=root_block,
            )
    return result


def _triton_fixed_wgrad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate the fixed tree for dY.T @ X in native [out, in] order."""
    m, tokens = a.shape
    n = b.size(1)
    result = a.new_empty((m, n))
    leaves = triton.cdiv(tokens, _BLOCK_K)
    # Unroll the first eight tree sizes, then specialize by tree depth.
    # Each depth has at most three group layouts (2, 3, or uneven leaves
    # per group) and two tail cases. For each case the output-feature
    # chunks are fixed, so specializations grow as O(log(tokens)).
    direct = leaves <= 8
    levels = 0 if direct else leaves.bit_length() - 2
    groups = 1 << levels
    group_leaves = leaves // groups if not direct and leaves % groups == 0 else 0
    has_tail = tokens % _BLOCK_K != 0
    # Two full leaves fit the wider row tile without increasing live ranges
    # across the extra leaves of an uneven or deeper subtree.
    block_m, block_n = (64 if group_leaves == 2 and not has_tail else 32), 64
    if direct:
        rows, columns = m, n
        workspace = result
    else:
        # Each group covers a canonical two- to four-leaf subtree. Bound the
        # scratch allocation and reuse it across output column chunks. Split
        # rows as well when all rows cannot fit one column tile.
        max_outputs = _MAX_TREE_WORKSPACE_ELEMENTS // groups
        rows = min(m, max(1, max_outputs // min(n, block_n)))
        column_limit = max_outputs // rows
        columns = min(
            n, column_limit if column_limit < block_n else column_limit // block_n * block_n
        )
        workspace = a.new_empty((groups, rows, columns))
    root_block = 32 if groups > 4096 else min(128, 4096 // groups)
    for row in range(0, m, rows):
        chunk_m = min(rows, m - row)
        lhs = a if row == 0 else a[row:]
        for col in range(0, n, columns):
            chunk_n = min(columns, n - col)
            tiles = triton.cdiv(chunk_m, block_m) * triton.cdiv(chunk_n, block_n)
            rhs = b if col == 0 else b[:, col:]
            grid = (tiles, groups) if levels <= 15 else (tiles * groups,)
            _fixed_wgrad_kernel[grid](
                lhs,
                rhs,
                workspace,
                chunk_m,
                tokens,
                chunk_n,
                *a.stride(),
                *b.stride(),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                DIRECT_LEAVES=leaves if direct else 0,
                GROUP_LEVELS=levels,
                HAS_TAIL=has_tail,
                GROUP_LEAVES=group_leaves,
                num_warps=4,
            )
            if not direct:
                _fixed_projection_root_kernel[(triton.cdiv(chunk_m * chunk_n, root_block),)](
                    workspace,
                    result[row:, col:],
                    chunk_m,
                    chunk_n,
                    n,
                    groups,
                    levels,
                    BLOCK=root_block,
                )
    return result


def _triton_tree_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    transpose_output: bool = False,
    out: torch.Tensor | None = None,
    preserve_a_strides: bool = False,
    inference_schedule: bool = False,
    fixed_leaf_tree: bool = False,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Triton deterministic GEMM expects two 2-D tensors")
    if a.size(1) != b.size(0):
        raise ValueError(f"Triton deterministic GEMM K mismatch: {a.size(1)} and {b.size(0)}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("Triton tree GEMM requires BF16 inputs")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise RuntimeError("Triton tree GEMM inputs must share one CUDA/ROCm device")

    # The leaf kernel accepts positive arbitrary strides. Wgrad uses this to
    # consume an activation transpose view without materializing another copy.
    # Other callers retain the established contiguous-input behavior.
    if not preserve_a_strides:
        a = a.contiguous()
    if not fixed_leaf_tree:
        b = b.contiguous()
    m_size, k_size = a.shape
    n_size = b.size(1)
    if fixed_leaf_tree and torch.version.hip is None and out is None and m_size > 0 and n_size > 0:
        if (
            transpose_output
            and k_size == 3 * _BLOCK_K
            and (weight_tiles := triton.cdiv(m_size, 64) * triton.cdiv(n_size, 64)) <= 2**31 - 1
        ):
            # (A @ B).T = B.T @ A.T. Tile dY.T @ X in native [out,in]
            # order so adjacent lanes write adjacent BF16 weight gradients.
            result = a.new_empty((n_size, m_size))
            _fixed_projection_leaves_kernel[(weight_tiles,)](
                b,
                a,
                result,
                n_size,
                m_size,
                b.stride(1),
                b.stride(0),
                a.stride(1),
                a.stride(0),
                BLOCK_M=64,
                BLOCK_N=64,
                WGRAD_96=True,
                PACK=8,
                STATIC_M=n_size if n_size in (256, 4096, 6656, 19968) else 0,
                num_warps=4,
            )
            return result
        if transpose_output and k_size > 0:
            # (A @ B).T = B.T @ A.T, both as stride-only views.
            return _triton_fixed_wgrad(b.T, a.T)
        if not transpose_output and k_size in _FIXED_PROJECTION_REDUCTIONS:
            return _triton_fixed_projection(a, b)
    plan = _device_tree_plan(k_size, a.device, fixed_leaf_tree=fixed_leaf_tree)
    workspace_elements = plan.host.node_count * m_size * n_size
    if workspace_elements > _MAX_TREE_WORKSPACE_ELEMENTS and out is None:
        rows_per_chunk = max(
            1,
            _MAX_TREE_WORKSPACE_ELEMENTS // (plan.host.node_count * n_size),
        )
        chunks = []
        for start in range(0, m_size, rows_per_chunk):
            stop = min(m_size, start + rows_per_chunk)
            chunk = _triton_tree_gemm(
                a[start:stop],
                b,
                transpose_output=transpose_output,
                preserve_a_strides=preserve_a_strides,
                inference_schedule=inference_schedule,
                fixed_leaf_tree=fixed_leaf_tree,
            )
            chunks.append(chunk)
        return torch.cat(chunks, dim=1 if transpose_output else 0)
    result_shape = (n_size, m_size) if transpose_output else (m_size, n_size)
    if out is None:
        result = torch.empty(result_shape, dtype=torch.bfloat16, device=a.device)
    else:
        if tuple(out.shape) != result_shape:
            raise ValueError(
                f"Triton tree GEMM output must have shape {result_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.bfloat16:
            raise TypeError(f"Triton tree GEMM output must be BF16, got {out.dtype}")
        if out.device != a.device:
            raise RuntimeError(f"Triton tree GEMM output must be on {a.device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError("Triton tree GEMM output buffer must be contiguous")
        if out.requires_grad:
            raise ValueError("Triton tree GEMM output buffer must not require gradients")
        result = out
    workspace = torch.empty(
        (plan.host.node_count, m_size, n_size),
        dtype=torch.bfloat16,
        device=a.device,
    )
    device_index = a.device.index if a.device.index is not None else torch.cuda.current_device()
    is_gfx942 = _device_arch(device_index) == "gfx942"
    use_inference_schedule = torch.is_inference_mode_enabled() or inference_schedule
    leaf_config = _tree_leaf_config(
        a.device,
        m_size,
        k_size,
        n_size,
        transpose_output=transpose_output,
        preserve_a_strides=preserve_a_strides,
    )
    if (
        _ROCM_TUNE_DECODE_LEAF
        and is_gfx942
        and use_inference_schedule
        and not transpose_output
        and not preserve_a_strides
    ):
        leaf_config = _gfx942_qwen_tp4_decode_leaf_config(m_size, k_size, n_size) or leaf_config
    tiles_m = triton.cdiv(m_size, leaf_config.block_m)
    tiles_n = triton.cdiv(n_size, leaf_config.block_n)
    leaf_grid = (
        (tiles_n, tiles_m, len(plan.host.leaf_nodes))
        if leaf_config.n_fastest
        else (len(plan.host.leaf_nodes), tiles_m, tiles_n)
    )
    direct_root_output = not fixed_leaf_tree and not transpose_output and is_gfx942
    fuse_leaf_reduction = (
        _ROCM_FUSE_LEAF_REDUCTION
        and direct_root_output
        and use_inference_schedule
        and k_size >= 1536
        and plan.rocm_leaf_reduction is not None
    )
    if fuse_leaf_reduction:
        assert plan.rocm_leaf_reduction is not None
        operation_count = len(plan.host.reduction_levels[0])
        fused_leaf_grid = (
            (tiles_n, tiles_m, operation_count)
            if leaf_config.n_fastest
            else (operation_count, tiles_m, tiles_n)
        )
        _det_gemm_tree_leaf_reduce_rocm_kernel[fused_leaf_grid](
            a,
            b,
            workspace,
            result,
            *plan.rocm_leaf_reduction,
            M=m_size,
            N=n_size,
            stride_am=a.stride(0),
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            stride_bn=b.stride(1),
            BLOCK_M=leaf_config.block_m,
            BLOCK_N=leaf_config.block_n,
            BLOCK_K=_BLOCK_K,
            N_FASTEST=leaf_config.n_fastest,
            WRITE_OUTPUT=len(plan.host.reduction_levels) == 1,
            num_warps=leaf_config.num_warps,
        )
    else:
        _det_gemm_kernel[leaf_grid](
            a,
            b,
            workspace,
            plan.leaf_starts,
            plan.leaf_lengths,
            plan.leaf_nodes,
            M=m_size,
            N=n_size,
            K=k_size,
            stride_am=a.stride(0),
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            stride_bn=b.stride(1),
            BLOCK_M=leaf_config.block_m,
            BLOCK_N=leaf_config.block_n,
            BLOCK_K=_BLOCK_K,
            N_FASTEST=leaf_config.n_fastest,
            FIXED_LEAVES=fixed_leaf_tree,
            num_warps=leaf_config.num_warps,
        )
    reduction_block = 256
    if fuse_leaf_reduction:
        blocks = triton.cdiv(m_size * n_size, reduction_block)
        for pair_index, nodes in enumerate(plan.rocm_fused_reduction_pairs_after_leaf):
            second_level = pair_index * 2 + 2
            operations = plan.host.reduction_levels[second_level]
            write_final_output = second_level == len(plan.host.reduction_levels) - 1
            _det_gemm_tree_reduce_two_levels_rocm_kernel[(len(operations), blocks)](
                workspace,
                result,
                *nodes,
                M=m_size,
                N=n_size,
                BLOCK=reduction_block,
                WRITE_OUTPUT=write_final_output,
            )
        remaining_levels = len(plan.host.reduction_levels) - 1
        if remaining_levels % 2:
            lower, upper, _output = plan.reduction_levels[-1]
            _det_gemm_tree_reduce_to_output_rocm_kernel[(blocks,)](
                workspace,
                result,
                lower,
                upper,
                M=m_size,
                N=n_size,
                BLOCK=reduction_block,
            )
    elif direct_root_output and use_inference_schedule:
        blocks = triton.cdiv(m_size * n_size, reduction_block)
        for pair_index, nodes in enumerate(plan.rocm_fused_reduction_pairs):
            second_level = pair_index * 2 + 1
            operations = plan.host.reduction_levels[second_level]
            write_final_output = second_level == len(plan.host.reduction_levels) - 1
            _det_gemm_tree_reduce_two_levels_rocm_kernel[(len(operations), blocks)](
                workspace,
                result,
                *nodes,
                M=m_size,
                N=n_size,
                BLOCK=reduction_block,
                WRITE_OUTPUT=write_final_output,
            )
        if len(plan.host.reduction_levels) % 2:
            lower, upper, _output = plan.reduction_levels[-1]
            _det_gemm_tree_reduce_to_output_rocm_kernel[(blocks,)](
                workspace,
                result,
                lower,
                upper,
                M=m_size,
                N=n_size,
                BLOCK=reduction_block,
            )
    else:
        for level_index, (operations, (lower, upper, output)) in enumerate(
            zip(
                plan.host.reduction_levels,
                plan.reduction_levels,
                strict=True,
            )
        ):
            write_final_output = (
                direct_root_output and level_index == len(plan.host.reduction_levels) - 1
            )
            if write_final_output and len(operations) != 1:
                raise RuntimeError("the final deterministic GEMM tree level must contain one root")
            if write_final_output:
                grid = (triton.cdiv(m_size * n_size, reduction_block),)
                # direct_root_output is true only for gfx942; CUDA retains its
                # established reduction and root-copy path.
                _det_gemm_tree_reduce_to_output_rocm_kernel[grid](
                    workspace,
                    result,
                    lower,
                    upper,
                    M=m_size,
                    N=n_size,
                    BLOCK=reduction_block,
                )
            else:
                blocks = triton.cdiv(m_size * n_size, reduction_block)
                # W[N,K] can exceed 65,535 blocks even after workspace chunking.
                # Move that dimension to grid X without changing any tree edges.
                blocks_on_x = blocks > 65535
                reduction_grid = (
                    (blocks, len(operations)) if blocks_on_x else (len(operations), blocks)
                )
                _det_gemm_tree_reduce_kernel[reduction_grid](
                    workspace,
                    lower,
                    upper,
                    output,
                    M=m_size,
                    N=n_size,
                    BLOCK=reduction_block,
                    BLOCKS_ON_X=blocks_on_x,
                )

    copy_block = 256
    if transpose_output:
        transpose_block = 32
        transpose_grid = (
            triton.cdiv(m_size, transpose_block),
            triton.cdiv(n_size, transpose_block),
        )
        _copy_tree_root_transposed_kernel[transpose_grid](
            workspace,
            result,
            plan.host.root,
            M=m_size,
            N=n_size,
            BLOCK_M=transpose_block,
            BLOCK_N=transpose_block,
        )
    elif not direct_root_output or not plan.host.reduction_levels:
        _copy_tree_root_kernel[(triton.cdiv(result.numel(), copy_block),)](
            workspace,
            result,
            plan.host.root,
            result.numel(),
            BLOCK=copy_block,
        )
    if out is not None:
        # Triton mutates caller-owned storage outside PyTorch's dispatcher.
        # Keep saved-tensor and cache version checks semantically correct.
        torch.autograd.graph.increment_version(result)
    return result


class _TritonDetGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, output_fp32=False):
        ctx.save_for_backward(a, b)
        ctx.output_fp32 = bool(output_fp32)
        return _triton_gemm_fp32(a, b) if output_fp32 else _triton_tree_gemm(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        da = (
            _triton_tree_gemm(grad_out.to(torch.bfloat16), b.t().contiguous()).reshape_as(a)
            if ctx.needs_input_grad[0]
            else None
        )
        db = (
            _triton_tree_gemm(a.t().contiguous(), grad_out.to(torch.bfloat16))
            if ctx.needs_input_grad[1]
            else None
        )
        record_backward(
            "det_gemm",
            kernel_id="rl_engine.backends.shared.triton.gemm.det_gemm._triton_gemm",
            impl="triton_det_gemm",
            family="triton",
        )
        return da, db, None


def _fixed_leaf_linear_forward(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    with torch.cuda.device(a.device):
        if (
            torch.version.hip is None
            and weight.ndim == 2
            and a.shape == (1, weight.size(1))
            and weight.size(1) in _FIXED_PROJECTION_REDUCTIONS
            and 0 < weight.size(0) <= 2**31 - 1
            and a.stride(1) == weight.stride(1) == 1
            and a.dtype == weight.dtype == torch.bfloat16
            and a.device == weight.device
        ):
            return _triton_fixed_decode(a, weight)
        return _triton_tree_gemm(a, weight.T, fixed_leaf_tree=True)


class _TritonDetLinearFn(torch.autograd.Function):
    """Fixed-leaf autograd for a native [N,K] weight."""

    @staticmethod
    def forward(ctx, a, weight):
        ctx.save_for_backward(a, weight)
        return _fixed_leaf_linear_forward(a, weight)

    @staticmethod
    def backward(ctx, grad_out):
        a, weight = ctx.saved_tensors
        da = dweight = None
        with torch.cuda.device(a.device):
            if ctx.needs_input_grad[0]:
                da = _triton_tree_gemm(grad_out, weight, fixed_leaf_tree=True)
            if ctx.needs_input_grad[1]:
                dweight = _triton_tree_gemm(
                    a.T,
                    grad_out,
                    transpose_output=True,
                    preserve_a_strides=True,
                    fixed_leaf_tree=True,
                )
        record_backward(
            "det_gemm",
            kernel_id="muse_glimmer.triton.fixed_leaf_tree",
            impl="muse_glimmer_dense_gemm",
            family="triton",
        )
        return da, dweight


def fixed_leaf_linear(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a native [N,K] weight through the fixed-leaf Triton kernels."""
    if torch.is_inference_mode_enabled():
        return _fixed_leaf_linear_forward(a, weight)
    return _TritonDetLinearFn.apply(a, weight)


class TritonDetGemmOp:
    """Batch-invariant deterministic GEMM, Triton path."""

    def __init__(self):
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available for TritonDetGemmOp")
        logger.info("TritonDetGemmOp ready (deterministic, autotune disabled).")

    def __call__(self, a, b):
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.device.type in ("cuda", "musa")
        assert b.device.type in ("cuda", "musa")
        assert a.device == b.device, "GEMM operands must be on the same device"
        return _TritonDetGemmFn.apply(a, b, False)

    def forward_fp32(self, a, b):
        if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("FP32-output Triton GEMM requires BF16 or FP32 inputs")
        assert a.device.type in ("cuda", "musa")
        assert b.device.type in ("cuda", "musa")
        assert a.device == b.device, "GEMM operands must be on the same device"
        return _TritonDetGemmFn.apply(a, b, True)

    forward_accum_fp32 = forward_fp32

    def parameter_vjp_contributions_fp32(self, *, a, b, grad_output):
        del b
        rows_a = a.float()
        rows_g = grad_output.float()
        return {"b": rows_a[:, :, None] * rows_g[:, None, :]}


def deterministic_gemm_triton(a, b):
    return _TritonDetGemmFn.apply(a, b, False)
