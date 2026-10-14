# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 projection through RL-Kernel's fixed FP32 Triton GEMM."""

import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import (
    _BLOCK_K,
    _TRITON_AVAILABLE,
    _triton_gemm_fp32,
    _triton_tree_gemm,
)
from rl_engine.ops.gemm.video_input_projection_fp32 import (
    VIDEO_PATCH_WIDTH,
    VideoInputProjectionFP32Op,
)


def projection_gemm_fp32(a, b):
    if a.shape[1] <= VIDEO_PATCH_WIDTH:
        return _triton_gemm_fp32(a, b)
    return _triton_tree_gemm(a, b, output_dtype=torch.float32, preserve_a_strides=True)


class TritonVideoInputProjectionFP32Op(VideoInputProjectionFP32Op):
    family = "triton_tree"
    backend_id = "triton-video-input-projection-fp32-v2"

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
            "kernel": "_det_gemm_fp32_kernel / _det_gemm_tree_leaf_kernel",
            "triton_version": triton.__version__,
            "tile": [64, 64, 32],
            "tf32": False,
            "fp_fusion": False,
            "split_k": 1,
            "reduction": "fixed_midpoint_tree",
            "reduction_leaf_size": _BLOCK_K,
            "reduction_backend": "triton",
            "leaf_launch": "all_leaves_in_one_grid",
        }
