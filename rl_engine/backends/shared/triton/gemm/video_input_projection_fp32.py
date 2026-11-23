# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 input projection through existing FP32 Triton GEMM paths.

The shared operator validates inputs and uses the canonical backward session
to collect and order parameter-gradient rows before their final reduction.
"""

import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import (
    _BLOCK_K,
    _TRITON_AVAILABLE,
    _triton_chunked_gemm_fp32,
    _triton_gemm_fp32,
    _triton_tree_gemm,
)
from rl_engine.ops.gemm.video_input_projection_fp32 import (
    VIDEO_PATCH_WIDTH,
    VideoInputProjectionFP32Op,
)

# H20-tuned output tile [M, N] and warp count for chunked dX/dW GEMM.
# Output tile dimensions are independent of the fixed K chunk width.
_DX_DW_BLOCK_M = 128
_DX_DW_BLOCK_N = 32
_DX_DW_NUM_WARPS = 4


def projection_gemm_fp32(a, b, *, bias=None):
    """Select a reused GEMM path from the fixed H3 input-projection dimensions."""

    # Forward has K=96; short parameter-gradient reductions also use full-K GEMM.
    if a.shape[1] <= VIDEO_PATCH_WIDTH:
        return _triton_gemm_fp32(a, b, bias=bias, aligned_tiles=True)
    # Long input and weight gradients both have 96 output columns.
    if b.shape[1] == VIDEO_PATCH_WIDTH:
        output = _triton_chunked_gemm_fp32(
            a, b, tile=(_DX_DW_BLOCK_M, _DX_DW_BLOCK_N, _DX_DW_NUM_WARPS)
        )
    else:
        # Bias gradients sum canonical token rows through the existing FP32 tree.
        output = _triton_tree_gemm(a, b, output_dtype=torch.float32, preserve_a_strides=True)
    return output if bias is None else output + bias


class TritonVideoInputProjectionFP32Op(VideoInputProjectionFP32Op):
    """Bind the shared H3 input-projection interface to CUDA SM90 Triton paths."""

    family = "triton"
    matmul_op = staticmethod(projection_gemm_fp32)
    backend_id = "triton-video-input-projection-fp32"

    def __init__(self):
        """Require Triton before constructing this backend."""

        if not _TRITON_AVAILABLE:
            raise ImportError("video input projection requires Triton")

    def _check_backend(self, x):
        """Enforce CUDA SM90 eligibility for the input tensor."""

        if x.device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("Triton video input projection is qualified on CUDA only")
        if torch.cuda.get_device_capability(x.device) != (9, 0):
            raise ValueError("video input projection requires CUDA SM90")

    def backend_info(self):
        """Describe the implementation configuration and scheduling policies.

        Scheduling fields describe available paths; input shapes determine
        which kernels execute for each call.
        """

        import triton

        return {
            "backend_id": self.backend_id,
            "family": "triton",
            "kernel": "_det_gemm_fp32_kernel / _det_gemm_fp32_chunk_merge_kernel / "
            "_det_gemm_kernel / "
            "_det_gemm_tree_leaf_reduce_rocm_kernel",
            "triton_version": triton.__version__,
            "tile": [64, 64, 32],
            "dx_dw_tile": [_DX_DW_BLOCK_M, _DX_DW_BLOCK_N, _BLOCK_K],
            "dx_dw_num_warps": _DX_DW_NUM_WARPS,
            "dx_dw_register_allocation": "compiler_default",
            "forward_aligned_tiles": True,
            "canonical_rows": "live_strict_order_proof_or_stable_sort",
            "tf32": False,
            "direct_gemm_fp_fusion": "compiler_default",
            "chunked_gemm_fp_fusion": False,
            "tree_gemm_fp_fusion": False,
            "dot_arithmetic": "ieee_fp32_simt_fma",
            "forward_bias": "fp32_epilogue",
            "leaf_loop_bound": "max_logical_leaf_length",
            "split_k": "fixed_128_term_dx_dw_chunks",
            "reduction": "fixed_128_chunks_balanced_fp32_merge",
            "dx_dw_chunk_k": 128,
            "dx_dw_chunk_accumulation": "two_64_term_halves",
            "dx_dw_merge_tile": 64,
            "bias_reduction": "fixed_midpoint_tree",
            "reduction_leaf_size": _BLOCK_K,
            "reduction_backend": "triton",
            "leaf_launch": "paired_leaves_for_one_row_complete_trees",
            "bias_tree_tile": [1, 128, 32],
            "tree_schedule": "paired_levels_full_workspace",
        }
