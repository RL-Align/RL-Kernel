# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA Qwen-Image latent pack/unpack (shared-memory tiled transpose)."""

import torch

from rl_engine.backends import extension
from rl_engine.ops.packing.latent_permutation import LatentKernelOp


class CudaLatentPackOp(LatentKernelOp):
    def __init__(self):
        if torch.version.hip is not None or not torch.cuda.is_available():
            raise RuntimeError("CUDA latent permutation requires an NVIDIA GPU")
        if not extension._EXT_AVAILABLE or not hasattr(extension._C, "latent_pack_unpack"):
            raise RuntimeError("Rebuild rl_engine._C with latent_pack_unpack.cu")
        self.kernel = extension._C.latent_pack_unpack


class CudaLatentUnpackOp(CudaLatentPackOp):
    unpack = True
