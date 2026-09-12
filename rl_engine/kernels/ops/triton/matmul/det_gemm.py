# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch- and TP-invariant deterministic GEMM, Triton path.

BF16 outputs use the same arithmetic graph as the native deterministic kernel:
each canonical K-tree leaf contains at most 32 values, accumulates in FP32, and
rounds to BF16.  Separate Triton kernels then evaluate the canonical midpoint
tree with BF16 nodes.  The strict ROCm leaf keeps the native gfx942 scalar FMA
order so both implementations have zero mismatch; a future MFMA leaf requires
a separately versioned arithmetic contract because its internal accumulation
does not match the scalar reference at every BF16 rounding boundary.

Autotuning and split-K are intentionally disabled.  FP32-output calls preserve
the earlier fixed-order, no-split-K contract and do not use BF16 tree nodes.
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

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.utils.logger import logger

# Pinned. NOT autotuned (autotune picks per-shape configs -> breaks invariance).
_BLOCK_M, _BLOCK_N, _BLOCK_K = 64, 64, 32
# Keep the intermediate tree workspace bounded.  A VLLM warmup can present a
# very large flattened token dimension; allocating ``node_count * M * N`` in
# one shot would otherwise create a multi-terabyte virtual tensor on ROCm.
# Chunking over M preserves the exact K-tree and BF16 rounding contract.
_MAX_TREE_WORKSPACE_ELEMENTS = 128 * 1024 * 1024
_TREE_PLANS: dict[tuple[int, int], "_DeviceTreePlan"] = {}
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
    if (
        transpose_output
        and preserve_a_strides
        and (m_size, n_size) in _QWEN_WGRAD_OUTPUT_SHAPES
    ):
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
    rocm_leaf_reduction: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None
    rocm_fused_reduction_pairs_after_leaf: tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...
    ]


def _build_tree_plan(k_size: int) -> _TreePlan:
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
        tuple(reductions_by_height.get(height, ()))
        for height in range(1, max_height + 1)
    )
    return _TreePlan(
        leaf_starts=tuple(leaf_starts),
        leaf_lengths=tuple(leaf_lengths),
        leaf_nodes=tuple(leaf_nodes),
        reduction_levels=reduction_levels,
        root=root,
        node_count=next_node,
    )


def _device_tree_plan(k_size: int, device: torch.device) -> _DeviceTreePlan:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device_index, k_size)
    with _TREE_PLAN_LOCK:
        cached = _TREE_PLANS.get(key)
        if cached is not None:
            return cached

        host = _build_tree_plan(k_size)

        def indices(values: tuple[int, ...]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        levels = []
        for operations in host.reduction_levels:
            lower, upper, output = zip(*operations, strict=True)
            levels.append((indices(lower), indices(upper), indices(output)))
        fused_pairs = []
        leaf_reduction = None
        fused_pairs_after_leaf = []
        if _device_arch(device_index) == "gfx942":
            def build_fused_pairs(start_level: int):
                result = []
                for level_index in range(
                    start_level, len(host.reduction_levels) - 1, 2
                ):
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
            acc += a[:, None] * b[None, :]
        output_offsets = (
            leaf_node * (M * N)
            + offs_m[:, None].to(tl.int64) * N
            + offs_n[None, :].to(tl.int64)
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
        result = lower.to(tl.float32) + upper_acc.to(
            workspace_ptr.dtype.element_ty
        ).to(tl.float32)
        output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        offsets = (
            offs_m[:, None].to(tl.int64) * N
            + offs_n[None, :].to(tl.int64)
        )
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
    ):
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
        value0 = tl.load(
            workspace_ptr + node0 * elements + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        value1 = tl.load(
            workspace_ptr + node1 * elements + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        value2 = tl.load(
            workspace_ptr + node2 * elements + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        value3 = tl.load(
            workspace_ptr + node3 * elements + offsets, mask=mask, other=0.0
        ).to(tl.float32)
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
            workspace_ptr
            + root * elements
            + offsets_m[:, None] * N
            + offsets_n[None, :],
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


def _triton_tree_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    transpose_output: bool = False,
    out: torch.Tensor | None = None,
    preserve_a_strides: bool = False,
    inference_schedule: bool = False,
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
    b = b.contiguous()
    m_size, k_size = a.shape
    n_size = b.size(1)
    plan = _device_tree_plan(k_size, a.device)
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
            )
            chunks.append(chunk)
        return torch.cat(chunks, dim=1 if transpose_output else 0)
    result_shape = (n_size, m_size) if transpose_output else (m_size, n_size)
    if out is None:
        result = torch.empty(result_shape, dtype=torch.bfloat16, device=a.device)
    else:
        if tuple(out.shape) != result_shape:
            raise ValueError(
                f"Triton tree GEMM output must have shape {result_shape}, "
                f"got {tuple(out.shape)}"
            )
        if out.dtype != torch.bfloat16:
            raise TypeError(f"Triton tree GEMM output must be BF16, got {out.dtype}")
        if out.device != a.device:
            raise RuntimeError(
                f"Triton tree GEMM output must be on {a.device}, got {out.device}"
            )
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
    use_inference_schedule = (
        torch.is_inference_mode_enabled() or inference_schedule
    )
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
        leaf_config = (
            _gfx942_qwen_tp4_decode_leaf_config(m_size, k_size, n_size)
            or leaf_config
        )
    tiles_m = triton.cdiv(m_size, leaf_config.block_m)
    tiles_n = triton.cdiv(n_size, leaf_config.block_n)
    leaf_grid = (
        (tiles_n, tiles_m, len(plan.host.leaf_nodes))
        if leaf_config.n_fastest
        else (len(plan.host.leaf_nodes), tiles_m, tiles_n)
    )
    direct_root_output = not transpose_output and is_gfx942
    fuse_leaf_reduction = (
        _ROCM_FUSE_LEAF_REDUCTION
        and direct_root_output
        and use_inference_schedule
        and k_size >= 1536
        and plan.rocm_leaf_reduction is not None
    )
    if fuse_leaf_reduction:
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
            num_warps=leaf_config.num_warps,
        )
    reduction_block = 256
    if fuse_leaf_reduction:
        blocks = triton.cdiv(m_size * n_size, reduction_block)
        for pair_index, nodes in enumerate(
            plan.rocm_fused_reduction_pairs_after_leaf
        ):
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
                raise RuntimeError(
                    "the final deterministic GEMM tree level must contain one root"
                )
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
                grid = (len(operations), triton.cdiv(m_size * n_size, reduction_block))
                _det_gemm_tree_reduce_kernel[grid](
                    workspace,
                    lower,
                    upper,
                    output,
                    M=m_size,
                    N=n_size,
                    BLOCK=reduction_block,
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
            kernel_id="rl_engine.kernels.ops.triton.matmul.det_gemm._triton_gemm",
            impl="triton_det_gemm",
            family="triton",
        )
        return da, db, None


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
