# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 output projection through the shared FP32 Triton GEMM."""

import torch

from rl_engine.backends.shared.triton.gemm.det_gemm import (
    _TRITON_AVAILABLE,
    _triton_chunked_gemm_fp32,
    _triton_gemm_fp32,
)
from rl_engine.ops.gemm.video_output_projection_fp32 import VideoOutputProjectionFP32Op

_PROJECTION_TILE = (128, 32, 4)


def projection_gemm_fp32(a, b, *, bias=None):
    """Reuse input-projection layouts for H3 output's FP32 matrix products.

    Forward reduces 5376 hidden channels. Weight and bias gradients reduce
    the session's canonical rows. dX reduces only 96 patch channels and keeps
    the original full-K GEMM. Tile selection is independent of physical rows.
    """
    rows, reduction = a.shape
    columns = b.shape[1]
    if reduction <= 96:
        output = _triton_gemm_fp32(a, b)
    elif rows == 96 and columns == 5376:
        # dW = G.T @ X. Compute X.T @ G, then restore [96, 5376]. This
        # matches input projection's dW layout and uses narrow output tiles.
        # Operands remain strided views; only the small final dW is copied.
        output = _triton_chunked_gemm_fp32(b.t(), a.t(), tile=_PROJECTION_TILE).t().contiguous()
    elif reduction == 5376 and columns == 96:
        # Forward matches input projection's dX layout. Keep this tile for
        # every row count, including the shape-identical dbias at K=5376.
        output = _triton_chunked_gemm_fp32(a, b, tile=_PROJECTION_TILE)
    else:
        output = _triton_chunked_gemm_fp32(a, b)
    return output if bias is None else output + bias


class TritonVideoOutputProjectionFP32Op(VideoOutputProjectionFP32Op):
    """Bind the H3 output-projection interface to CUDA SM90 Triton GEMM."""

    family = "triton"
    matmul_op = staticmethod(projection_gemm_fp32)
    backend_id = "triton-video-output-projection-fp32"

    def __init__(self):
        if not _TRITON_AVAILABLE:
            raise ImportError("video output projection requires Triton")

    def _check_backend(self, x):
        """Enforce the CUDA SM90 qualification for the input tensor."""

        if x.device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("Triton video output projection is qualified on CUDA only")
        if torch.cuda.get_device_capability(x.device) != (9, 0):
            raise ValueError("video output projection requires CUDA SM90")

    def backend_info(self):
        """Describe implementation configuration, rather than a call trace."""

        import triton

        return {
            "backend_id": self.backend_id,
            "family": self.family,
            "kernel": "_det_gemm_fp32_kernel / _det_gemm_fp32_chunk_merge_kernel",
            "chunk_k": 128,
            "reduction": "fixed_128_chunks_balanced_fp32_merge",
            "projection_tile": _PROJECTION_TILE,
            "weight_gradient": "transposed_product_when_k_gt_96",
            "triton_version": triton.__version__,
            "canonical_rows": "live_strict_order_proof_or_stable_sort",
            "tf32": False,
        }
