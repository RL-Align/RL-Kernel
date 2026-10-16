# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 projection through RL-Kernel's fixed FP32 Triton GEMM."""

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

_VJP_TILE = (128, 32, 4)


def projection_gemm_fp32(a, b, *, bias=None):
    if a.shape[1] <= VIDEO_PATCH_WIDTH:
        return _triton_gemm_fp32(a, b, bias=bias)
    if b.shape[1] == VIDEO_PATCH_WIDTH:
        output = _triton_chunked_gemm_fp32(a, b, tile=_VJP_TILE)
    else:
        output = _triton_tree_gemm(a, b, output_dtype=torch.float32, preserve_a_strides=True)
    return output if bias is None else output + bias


class TritonVideoInputProjectionFP32Op(VideoInputProjectionFP32Op):
    family = "triton_tree"
    backend_id = "triton-video-input-projection-fp32-v7"

    def __init__(self):
        if not _TRITON_AVAILABLE:
            raise ImportError("video input projection requires Triton")

    def _check_backend(self, x):
        if x.device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("Triton video input projection is qualified on CUDA only")
        if torch.cuda.get_device_capability(x.device) != (9, 0):
            raise ValueError("video input projection requires CUDA SM90")

    def backend_info(self):
        import triton

        return {
            "backend_id": self.backend_id,
            "family": "triton",
            "kernel": "_det_gemm_fp32_kernel / _det_gemm_fp32_chunk_merge_kernel / "
            "_det_gemm_kernel / "
            "_det_gemm_tree_leaf_reduce_rocm_kernel",
            "triton_version": triton.__version__,
            "tile": [64, 64, 32],
            "vjp_tile": [_VJP_TILE[0], _VJP_TILE[1], _BLOCK_K],
            "vjp_num_warps": _VJP_TILE[2],
            "tf32": False,
            "fp_fusion": False,
            "dot_arithmetic": "ieee_fp32_simt_fma",
            "forward_bias": "fp32_epilogue",
            "leaf_loop_bound": "max_logical_leaf_length",
            "split_k": "fixed_128_term_vjp_chunks",
            "reduction": "fixed_128_chunks_balanced_fp32_merge",
            "vjp_chunk_k": 128,
            "vjp_merge_tile": 64,
            "bias_reduction": "fixed_midpoint_tree",
            "reduction_leaf_size": _BLOCK_K,
            "reduction_backend": "triton",
            "leaf_launch": "paired_leaves_for_one_row_complete_trees",
            "tree_tile": [64, 32, 32],
            "bias_tree_tile": [1, 128, 32],
            "tree_schedule": "paired_levels_compact_workspace",
        }
