# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 projection through RL-Kernel's fixed FP32 Triton GEMM."""

import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import _TRITON_AVAILABLE
from rl_engine.ops.gemm.video_input_projection_fp32 import VideoInputProjectionFP32Op


class TritonVideoInputProjectionFP32Op(VideoInputProjectionFP32Op):
    family = "triton"
    backend_id = "triton-video-input-projection-fp32-v1"

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
            "family": self.family,
            "kernel": "_det_gemm_fp32_kernel",
            "triton_version": triton.__version__,
            "tile": [64, 64, 32],
            "tf32": False,
            "fp_fusion": False,
            "split_k": 1,
            "reduction": "fixed_midpoint_tree",
            "reduction_leaf_size": 128,
        }
