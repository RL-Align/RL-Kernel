# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
import torch

from rl_engine.kernels.ops import base
from rl_engine.kernels.ops.latent_layout import LatentKernelOp


class CudaLatentPackOp(LatentKernelOp):
    def __init__(self):
        if torch.version.hip is not None or not torch.cuda.is_available():
            raise RuntimeError("CUDA latent permutation requires an NVIDIA GPU")
        if not base._EXT_AVAILABLE or not hasattr(base._C, "latent_pack_unpack"):
            raise RuntimeError("Rebuild rl_engine._C with latent_pack_unpack.cu")
        self.kernel = base._C.latent_pack_unpack


class CudaLatentUnpackOp(CudaLatentPackOp):
    unpack = True
