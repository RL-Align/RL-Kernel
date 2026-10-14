# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 FP32 projection using a pinned CUTLASS SIMT GEMM candidate.

CUTLASS dependency notice (BSD-3-Clause):
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from functools import lru_cache
from importlib import import_module

import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import (
    _BLOCK_K,
    _TRITON_AVAILABLE,
    _triton_tree_gemm,
)
from rl_engine.ops.gemm.video_input_projection_fp32 import (
    VIDEO_PATCH_WIDTH,
    VideoInputProjectionFP32Op,
)

CUTLASS_REVISION = "56ca28adb32ffadd097566545b8f7d83b652e998"
CUTLASS_HEADERS_SHA256 = "7b5321860c269f8e44eab02e59c32c2d4d0e2fa7a3ef72e0a161de9f58b361fa"


@lru_cache(maxsize=1)
def _load_extension():
    try:
        extension = import_module("rl_engine._C_cutlass")
    except ImportError as exc:
        raise RuntimeError(
            "build the optional CUTLASS extension with RL_KERNEL_BUILD_CUTLASS=1"
        ) from exc
    info = extension.build_info()
    if (
        info["cutlass_revision"] != CUTLASS_REVISION
        or info["headers_sha256"] != CUTLASS_HEADERS_SHA256
        or info["fmad"]
        or info["ftz"]
        or info["split_k"] != 1
        or info.get("tree_leaf_size") != _BLOCK_K
    ):
        raise RuntimeError("CUTLASS extension does not match the strict FP32 build contract")
    return extension


def _cutlass_tree_leaves(a, b, workspace, plan):
    _load_extension().gemm_fp32_tree_leaves_tn(
        a, b.t(), workspace, plan.leaf_starts, plan.leaf_lengths, plan.leaf_nodes
    )


def cutlass_gemm_fp32(a, b):
    # The native TN interface represents mathematical B[K,N] as [N,K] storage.
    if a.shape[1] <= VIDEO_PATCH_WIDTH:
        return _load_extension().gemm_fp32_tn(a.contiguous(), b.t().contiguous())
    return _triton_tree_gemm(
        a,
        b,
        output_dtype=torch.float32,
        preserve_a_strides=True,
        fp32_leaf_launcher=_cutlass_tree_leaves,
    )


class CUDAVideoInputProjectionFP32Op(VideoInputProjectionFP32Op):
    family = "cutlass"
    backend_id = "cuda-cutlass-video-input-projection-fp32-v2"

    def __init__(self):
        if not _TRITON_AVAILABLE:
            raise RuntimeError("CUTLASS projection requires Triton for shared FP32 tree reductions")
        _load_extension()

    def _check_backend(self, x):
        if x.device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("CUTLASS video input projection requires CUDA")
        if torch.cuda.get_device_capability(x.device) != (9, 0):
            raise ValueError("CUTLASS video input projection requires CUDA SM90")

    def backend_info(self):
        import triton

        return {
            "backend_id": self.backend_id,
            "family": "cuda",
            "kernel": "gemm_fp32_tn / gemm_fp32_tree_leaves_tn",
            "reduction": "fixed_midpoint_tree",
            "reduction_leaf_size": _BLOCK_K,
            "reduction_backend": "triton",
            "triton_version": triton.__version__,
            "leaf_launch": "cutlass_device_gemm_array",
            **_load_extension().build_info(),
        }
